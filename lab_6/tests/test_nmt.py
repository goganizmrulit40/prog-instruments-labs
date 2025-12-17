import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import EncoderRNN
from models import DecoderRNN
from dataloader import load_dataset
from models import DecoderAttenRNN
import torch.nn.functional as F
from models import SimpleNMT



def test_encoder_rnn_basic():
    """Тест 1: Базовая проверка EncoderRNN"""
    vocab_size = 50
    hidden_size = 64
    batch_size = 4
    seq_length = 10

    encoder = EncoderRNN(vocab_size=vocab_size, hidden_size=hidden_size)

    assert encoder.hidden_size == hidden_size
    assert isinstance(encoder.gru, nn.GRU)

    assert encoder.gru.input_size == vocab_size
    assert encoder.gru.hidden_size == hidden_size
    assert encoder.gru.batch_first == True

    test_input = torch.randn(batch_size, seq_length, vocab_size)
    init_hidden = torch.zeros(1, batch_size, hidden_size)

    seq_output, last_state = encoder(test_input, init_hidden)

    assert isinstance(seq_output, torch.Tensor)
    assert isinstance(last_state, torch.Tensor)
    assert seq_output.shape == (batch_size, seq_length, hidden_size)
    assert last_state.shape == (1, batch_size, hidden_size)


def test_decoder_rnn_without_attention():
    """Тест 2: Декодер без механизма внимания"""
    vocab_size = 40
    hidden_size = 64
    output_size = 30
    batch_size = 3
    seq_length = 8

    decoder = DecoderRNN(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        output_size=output_size
    )

    assert isinstance(decoder.gru, nn.GRU)
    assert isinstance(decoder.hidden2index, nn.Linear)
    assert decoder.hidden2index.out_features == output_size

    decoder_input = torch.randn(batch_size, seq_length, vocab_size)
    init_state = torch.randn(1, batch_size, hidden_size)

    seq_output, last_state = decoder(decoder_input, init_state)

    assert seq_output.shape == (batch_size, seq_length, output_size)
    assert last_state.shape == (1, batch_size, hidden_size)

    predictions = seq_output.argmax(dim=-1)
    assert torch.all(predictions >= 0) and torch.all(predictions < output_size)


@pytest.mark.parametrize("dataset_size,min_expected", [
    (1, 0),
    (5, 3),
    (10, 7),
    (20, 15),
])
def test_load_dataset_parametrized(dataset_size, min_expected):
    """Тест 3: Параметризованная проверка загрузки датасета"""
    dataset = load_dataset(dataset_size)

    assert isinstance(dataset, list)
    assert len(dataset) <= dataset_size

    if dataset_size > 0:
        assert len(dataset) >= min_expected or len(dataset) == 0

    if dataset:
        for human_date, machine_date in dataset:
            assert isinstance(human_date, str)
            assert isinstance(machine_date, str)
            assert human_date == human_date.lower()
            assert ',' not in human_date

            parts = machine_date.split('-')
            assert len(parts) == 3
            assert all(part.isdigit() for part in parts)


def test_attention_mechanism():
    """Тест 4: Проверка механизма внимания"""
    vocab_size = 40
    hidden_size = 64
    output_size = 30

    decoder = DecoderAttenRNN(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        output_size=output_size
    )

    batch_size = 2
    seq_length = 7

    decoder_hidden = torch.randn(1, batch_size, hidden_size)
    encoder_output = torch.randn(batch_size, seq_length, hidden_size)

    context_vector = decoder.calculate_attention(decoder_hidden, encoder_output)

    assert context_vector.shape == (batch_size, hidden_size)

    decoder_hidden_permuted = decoder_hidden.permute(1, 2, 0)
    scores = torch.bmm(encoder_output, decoder_hidden_permuted).squeeze(2)
    attention_weights = F.softmax(scores, dim=1).unsqueeze(2)

    weights_sum = attention_weights.sum(dim=1)
    assert torch.allclose(weights_sum, torch.ones(batch_size, 1), rtol=1e-5)
    assert torch.all(attention_weights >= 0) and torch.all(attention_weights <= 1)


def test_simplenmt_training_mode():
    """Тест 5: SimpleNMT в режиме обучения"""
    for with_attention in [False, True]:
        model = SimpleNMT(
            in_vocab_size=60,
            out_vocab_size=50,
            in_hidden_size=64,
            out_hidden_size=64,
            output_size=50,
            with_attention=with_attention
        )

        assert model.with_attention == with_attention

        batch_size = 3
        encoder_input = torch.randn(batch_size, 8, 60)
        encoder_init_hidden = torch.zeros(1, batch_size, 64)
        decoder_input = torch.randn(batch_size, 6, 50)

        logits = model(
            encoder_input=encoder_input,
            encoder_init_hidden=encoder_init_hidden,
            decoder_input=decoder_input
        )

        assert logits.shape == (batch_size, 6, 50)
