import pytest
from unittest.mock import patch
import main


def test_hangman_begin():
    """Тест начального состояния виселицы"""
    result = main.hangman(8)
    assert "O" not in result
    assert "|      |" in result


def test_hangman_end():
    """Тест финального состояния"""
    result = main.hangman(0)
    assert "O" in result
    assert "\\|/" in result


def test_get_word_easy():
    """Тест выбора слова для лёгкого уровня"""
    with patch('main.DIFFICULTY_LEVEL', 'easy'), \
         patch('random.choice', lambda lst: lst[0]):
        assert main.get_word() == "EGG"


def test_getword_middle():
    """Тест выбора слова для среднего уровня"""
    with patch('main.DIFFICULTY_LEVEL', 'medium'), \
         patch('random.choice', lambda lst: lst[0]):
        assert main.get_word() == "SOUP"