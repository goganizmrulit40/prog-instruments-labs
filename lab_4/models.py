import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.5):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(vocab_size, hidden_size, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, init_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_output, last_state = self.gru(x, init_hidden)
        return seq_output, last_state


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, output_size: int, dropout: float = 0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(vocab_size, hidden_size, dropout=dropout, batch_first=True)
        self.hidden2index = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, init_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_output, last_state = self.gru(x, init_state)
        seq_output = self.hidden2index(seq_output)
        return seq_output, last_state


class DecoderAttenRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, output_size: int, dropout: float = 0.5):
        super(DecoderAttenRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(vocab_size, hidden_size, dropout=dropout, batch_first=True)
        self.hidden2label = nn.Linear(hidden_size, output_size)
        self.atten_affine = nn.Linear(hidden_size*2, hidden_size)

    def calculate_attention(self, decoder_hidden: torch.Tensor,
                            encoder_output: torch.Tensor) -> torch.Tensor:
        # Транспонирование для матричного умножения
        decoder_hidden = decoder_hidden.permute(1, 2, 0)   # (batch_size, hidden_size, 1)
        # Вычисление баллов внимания
        scores = torch.bmm(encoder_output, decoder_hidden).squeeze(2)  # (batch_size, seq_len)
        # Применение softmax для получения весов внимания
        attention_weights = F.softmax(scores, dim=1).unsqueeze(2)       # (batch_size, seq_len, 1)
        # Вычисление контекстного вектора (взвешенная сумма)
        context_vector = (attention_weights * encoder_output).sum(dim=1)    # (batch_size, hidden_size)

        return context_vector

    def forward(self, x: torch.Tensor, init_state: torch.Tensor,
                seq_encoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_len, _ = x.shape
        current_hidden = init_state
        decoder_outputs = []

        for i in range(max_len):
            # Вычисление attention вектора
            attention_context = self.calculate_attention(current_hidden, seq_encoder_output)

            # Объединение attention контекста с текущим скрытым состоянием
            combined = torch.cat([attention_context.unsqueeze(0), current_hidden], dim=2)
            combined = self.atten_affine(combined)

            # Пропуск через GRU
            output, current_hidden = self.gru(x[:, i, :].unsqueeze(1), combined)

            # Преобразование в выходное пространство
            output = self.hidden2label(output.squeeze(1))
            decoder_outputs.append(output)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return decoder_outputs, current_hidden


class SimpleNMT(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int,
                 in_hidden_size: int, out_hidden_size: int,
                 output_size: int, with_attention: bool = False):
        super(SimpleNMT, self).__init__()
        self.with_attention = with_attention
        self.encoder = EncoderRNN(in_vocab_size, in_hidden_size)
        if self.with_attention:
            self.decoder = DecoderAttenRNN(out_vocab_size, out_hidden_size, output_size)
        else:
            self.decoder = DecoderRNN(out_vocab_size, out_hidden_size, output_size)

    def _training_forward(self, encoder_input: torch.Tensor, encoder_init_hidden: torch.Tensor,
                          decoder_input: torch.Tensor, encoder_seq_output: torch.Tensor,
                          encoder_last_state: torch.Tensor) -> torch.Tensor:
        """Прямой проход в режиме обучения"""
        if self.with_attention:
            logits, _ = self.decoder(decoder_input, encoder_last_state, encoder_seq_output)
        else:
            logits, _ = self.decoder(decoder_input, encoder_last_state)
        return logits

    def _inference_forward(self, encoder_seq_output: torch.Tensor, encoder_last_state: torch.Tensor,
                           out_word2index: Dict[str, int], out_index2word: Dict[int, str],
                           max_len: int, out_size: int) -> List[List[str]]:
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

    def _decode_single_sequence(self, encoder_output_single: torch.Tensor,
                                hidden_state_single: torch.Tensor,
                                out_word2index: Dict[str, int], out_index2word: Dict[int, str],
                                max_len: int, out_size: int) -> List[str]:
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

    def _decoder_step(self, decoder_input: torch.Tensor, hidden_state: torch.Tensor,
                      encoder_output: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Один шаг декодера"""
        if self.with_attention:
            return self.decoder(decoder_input, hidden_state, encoder_output)
        else:
            return self.decoder(decoder_input, hidden_state)

    def _create_start_token(self, out_word2index: Dict[str, int], out_size: int) -> torch.Tensor:
        """Создание начального токена"""
        return torch.FloatTensor(
            np.eye(out_size)[[out_word2index["<start>"]]]
        ).unsqueeze(0)

    def _create_next_token(self, token_idx: int, out_size: int) -> torch.Tensor:
        """Создание следующего токена"""
        return torch.FloatTensor(
            [np.eye(out_size)[token_idx]]
        ).unsqueeze(0)

    def forward(self, encoder_input: torch.Tensor, encoder_init_hidden: torch.Tensor,
                decoder_input: Optional[torch.Tensor] = None,
                out_word2index: Optional[Dict[str, int]] = None,
                out_index2word: Optional[Dict[int, str]] = None,
                max_len: Optional[int] = None,
                out_size: Optional[int] = None) -> Union[torch.Tensor, List[List[str]]]:
        encoder_seq_output, encoder_last_state = self.encoder(encoder_input, encoder_init_hidden)

        if decoder_input is not None:
            return self._training_forward(encoder_input, encoder_init_hidden, decoder_input,
                                          encoder_seq_output, encoder_last_state)
        else:
            return self._inference_forward(encoder_seq_output, encoder_last_state,
                                           out_word2index, out_index2word, max_len, out_size)