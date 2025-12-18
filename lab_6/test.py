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