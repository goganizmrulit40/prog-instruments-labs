import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import EncoderRNN


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
