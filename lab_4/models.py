import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(vocab_size, hidden_size, dropout=dropout, batch_first=True)

    def forward(self, x, init_hidden):
        seq_output, last_state = self.gru(x, init_hidden)
        return seq_output, last_state


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(vocab_size, hidden_size, dropout=dropout, batch_first=True)
        self.hidden2index = nn.Linear(hidden_size, output_size)

    def forward(self, x, init_state):
        seq_output, last_state = self.gru(x, init_state)
        seq_output = self.hidden2index(seq_output)
        return seq_output, last_state


class DecoderAttenRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, dropout=0.5):
        super(DecoderAttenRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(vocab_size, hidden_size, dropout=dropout, batch_first=True)
        self.hidden2label = nn.Linear(hidden_size, output_size)
        self.atten_affine = nn.Linear(hidden_size*2, hidden_size)

    def get_alpha(self, hi, encoder_output):
        # hi shape (1, batch_size, hidden_size)
        # encoder_output (batch, seq_len, hidden_size)
        hi = hi.permute(1, 2, 0)   # (batch_size, hidden_size, 1)
        # print(encoder_output.shape, hi.shape)
        e = torch.bmm(encoder_output, hi).squeeze(2)  # (batch_size, seq_len)
        e = F.softmax(e, dim=1).unsqueeze(2)       # (batch_size, seq_len, 1)
        alpha = (e * encoder_output).sum(dim=1)    # (batch_size, hidden_size)

        return alpha

    def forward(self, x, init_state, seq_encoder_output):
        # print(x.shape, init_state.shape, seq_encoder_output.shape)
        batch_size, max_len, _ = x.shape  # 独热码表示
        hi = init_state
        seq_decoder_output = []
        for i in range(max_len):
            # alpha shape (batch_size, hidden_size)
            alpha = self.get_alpha(hi, seq_encoder_output)  # alpha 表示当前time step的隐状态矩阵和encoder的各个time step输出的关联
            hi = torch.cat([alpha.unsqueeze(0), hi], dim=2)
            hi = self.atten_affine(hi)
            output, hi = self.gru(x[:, i, :].unsqueeze(1), hi)
            seq_output = self.hidden2label(output.squeeze(1))
            seq_decoder_output.append(seq_output.squeeze(1))
        seq_decoder_output = torch.stack(seq_decoder_output, dim=1)
        return seq_decoder_output, hi


class SimpleNMT(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, in_hidden_size, out_hidden_size, output_size, with_attention=False):
        super(SimpleNMT, self).__init__()
        self.with_attention = with_attention
        self.encoder = EncoderRNN(in_vocab_size, in_hidden_size)
        if self.with_attention:
            self.decoder = DecoderAttenRNN(out_vocab_size, out_hidden_size, output_size)
        else:
            self.decoder = DecoderRNN(out_vocab_size, out_hidden_size, output_size)

    def _training_forward(self, encoder_input, encoder_init_hidden, decoder_input, encoder_seq_output,
                          encoder_last_state):
        """Прямой проход в режиме обучения"""
        if self.with_attention:
            logits, _ = self.decoder(decoder_input, encoder_last_state, encoder_seq_output)
        else:
            logits, _ = self.decoder(decoder_input, encoder_last_state)
        return logits

    def _inference_forward(self, encoder_seq_output, encoder_last_state, out_word2index, out_index2word, max_len,
                           out_size):
        """Прямой проход в режиме инференса"""
        decoded_sents = []
        for i in range(len(encoder_seq_output)):
            sent = self._decode_single_sequence(
                encoder_seq_output[i],
                encoder_last_state[:, i],
                out_word2index,
                out_index2word,
                max_len,
                out_size
            )
            decoded_sents.append(sent)
        return decoded_sents

    def _decode_single_sequence(self, encoder_output_single, hidden_state_single,
                                out_word2index, out_index2word, max_len, out_size):
        """Декодирование одной последовательности"""
        sent = []
        decoder_input = self._create_start_token(out_word2index, out_size)
        hi = hidden_state_single.unsqueeze(1)

        for _ in range(max_len):
            decoder_output, hi = self._decoder_step(decoder_input, hi, encoder_output_single.unsqueeze(0))
            token_idx = decoder_output.data.argmax(1).item()

            if token_idx == out_word2index["<end>"]:
                break

            sent.append(out_index2word[token_idx])
            decoder_input = self._create_next_token(token_idx, out_size)

        return sent

    def _decoder_step(self, decoder_input, hidden_state, encoder_output=None):
        """Один шаг декодера"""
        if self.with_attention:
            return self.decoder(decoder_input, hidden_state, encoder_output)
        else:
            return self.decoder(decoder_input, hidden_state)

    def _create_start_token(self, out_word2index, out_size):
        """Создание начального токена"""
        return torch.FloatTensor(
            np.eye(out_size)[[out_word2index["<start>"]]]
        ).unsqueeze(0)

    def _create_next_token(self, token_idx, out_size):
        """Создание следующего токена"""
        return torch.FloatTensor(
            [np.eye(out_size)[token_idx]]
        ).unsqueeze(0)

    def forward(self, encoder_input, encoder_init_hidden, decoder_input=None,
                out_word2index=None, out_index2word=None, max_len=None, out_size=None):
        encoder_seq_output, encoder_last_state = self.encoder(encoder_input, encoder_init_hidden)

        if decoder_input is not None:
            return self._training_forward(encoder_input, encoder_init_hidden, decoder_input,
                                          encoder_seq_output, encoder_last_state)
        else:
            return self._inference_forward(encoder_seq_output, encoder_last_state,
                                           out_word2index, out_index2word, max_len, out_size)