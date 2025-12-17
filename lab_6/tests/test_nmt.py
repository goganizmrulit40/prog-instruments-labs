import pytest
import torch
import torch.nn as nn
import sys
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models import EncoderRNN, DecoderRNN, DecoderAttenRNN, SimpleNMT
from dataloader import load_dataset, prepare_data
import torch.nn.functional as F
from unittest.mock import patch, Mock, mock_open, MagicMock


def test_encoder_rnn_basic():
    """Тест 1: Базовая проверка EncoderRNN.
    Тестирует инициализацию и прямой проход кодировщика."""
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
    """Тест 2: Декодер без механизма внимания.
    Тестирует архитектуру и выходные данные DecoderRNN."""
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
    """Тест 3: Параметризованная проверка загрузки датасета.
    Проверяет работу функции с разными размерами входных данных.
    Каждый набор параметров запускается как отдельный тест."""
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
    """Тест 4: Проверка механизма внимания.
    Проверяет расчет весов внимания и контекстного вектора."""
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


def test_simple_nmt_training_mode():
    """Тест 5: SimpleNMT в режиме обучения.
    Проверяет обе версии модели: с механизмом внимания и без."""
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


def test_simple_nmt_inference_with_mocks():
    """Тест 6: SimpleNMT в режиме инференса с моками.
    Использует заглушки для словарей и методов для изоляции теста."""
    model = SimpleNMT(
        in_vocab_size=60,
        out_vocab_size=50,
        in_hidden_size=64,
        out_hidden_size=64,
        output_size=50,
        with_attention=True
    )

    mock_word2index = {
        "<start>": 0,
        "<end>": 1,
        "1980": 2,
        "05": 3,
        "10": 4
    }
    mock_index2word = {v: k for k, v in mock_word2index.items()}

    encoder_input = torch.randn(2, 8, 60)
    encoder_init_hidden = torch.zeros(1, 2, 64)

    with patch.object(model, '_inference_forward') as mock_inference:
        mock_inference.return_value = [["1980", "05", "10"], ["1999", "12", "25"]]

        result = model(
            encoder_input=encoder_input,
            encoder_init_hidden=encoder_init_hidden,
            out_word2index=mock_word2index,
            out_index2word=mock_index2word,
            max_len=10,
            out_size=50
        )

        mock_inference.assert_called_once()
        assert isinstance(result, list)
        assert len(result) == 2

    start_token = model._create_start_token(mock_word2index, 50)
    assert start_token.shape == (1, 1, 50)

    next_token = model._create_next_token(2, 50)
    assert next_token.shape == (1, 1, 50)


@patch('os.path.exists')
@patch('pandas.read_csv')
@patch('dataloader.load_dataset')
@patch('pandas.DataFrame.to_csv')
def test_prepare_data_with_mocks(mock_to_csv, mock_load_dataset,
                                 mock_read_csv, mock_exists):
    """Тест 7: Подготовка данных с моками.
    Проверяет создание файлов и работу в разных режимах (debug=True/False)."""
    mock_exists.return_value = False

    test_data = [
        ["monday may 7 1983", "1983-05-07"],
        ["19 march 1998", "1998-03-19"],
        ["18 jul 2008", "2008-07-18"]
    ]
    mock_load_dataset.return_value = test_data

    mock_read_csv.return_value = pd.DataFrame()

    train_file, eval_file = prepare_data(
        dataset_path="test_path",
        dataset_size=10,
        debug=True
    )

    mock_load_dataset.assert_called_once_with(10)
    assert mock_to_csv.call_count == 2

    assert isinstance(train_file, str)
    assert isinstance(eval_file, str)
    assert "train_small.csv" in train_file
    assert "eval_small.csv" in eval_file

    mock_load_dataset.reset_mock()
    mock_to_csv.reset_mock()

    train_file, eval_file = prepare_data(
        dataset_path="test_path",
        dataset_size=100,
        debug=False
    )

    mock_load_dataset.assert_called_once_with(100)
    assert "train.csv" in train_file
    assert "eval.csv" in eval_file
