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


@pytest.mark.parametrize("user_input,expected_tries,expected_global_value", [
    ("easy", 8, "easy"),
    ("medium", 6, "medium"),
    ("hard", 4, "hard"),
    ("EASY", 8, "easy"),  # case-insensitivity
    ("Medium", 6, "medium"),  # mixed case
    ("invalid", 6, "invalid"),  # невалидный ввод
    ("", 6, ""),  # пустой ввод
])
def test_choose_difficulty_parametrized(user_input, expected_tries, expected_global_value, capsys):
    """
    Параметризованный тест для функции choose_difficulty.
    Проверяет:
    1. Возвращает правильное количество попыток
    2. Корректно обновляет глобальную переменную DIFFICULTY_LEVEL
    3. Обрабатывает разные регистры ввода
    4. Обрабатывает невалидный ввод (сообщение в консоль)
    """
    with patch('builtins.input', return_value=user_input):
        original_difficulty = main.DIFFICULTY_LEVEL

        result = main.choose_difficulty()

        assert result == expected_tries

        assert main.DIFFICULTY_LEVEL == expected_global_value

        if user_input.lower() not in ['easy', 'medium', 'hard'] and user_input != "":
            captured = capsys.readouterr()
            assert "Invalid difficulty level. Defaulting to medium." in captured.out

        main.DIFFICULTY_LEVEL = original_difficulty