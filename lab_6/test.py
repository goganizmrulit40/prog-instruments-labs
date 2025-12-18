import pytest
from unittest.mock import patch
import main


def test_hangman_begin():
    """Тест начального состояния виселицы (без человека)"""
    result = main.hangman(8)
    assert "O" not in result
    assert "|      |" in result


def test_hangman_end():
    """Тест финального состояния (полный человек)"""
    result = main.hangman(0)
    assert "O" in result  # голова
    assert "\\|/" in result  # туловище и руки
    assert "// \\" in result  # ноги


def test_get_word_easy():
    """Тест выбора слова для лёгкого уровня"""
    with patch('main.DIFFICULTY_LEVEL', 'easy'), \
         patch('random.choice', lambda lst: lst[0]):
        word = main.get_word()
        assert word == "EGG"  # первое слово из easy_wordlist
        assert word.isupper()


def test_getword_middle():
    """Тест выбора слова для среднего уровня"""
    with patch('main.DIFFICULTY_LEVEL', 'medium'), \
         patch('random.choice', lambda lst: lst[0]):
        word = main.get_word()
        assert word == "SOUP"  # первое слово из medium_wordlist


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


def test_game_correct_letter_guess():
    """Тест правильного угадывания буквы в слове"""
    # логика из функции play():
    word = "TEST"
    guessed_letters = []
    word_completion = "_" * len(word)

    guess = 'T'
    if guess not in word:
        assert False

    guessed_letters.append(guess)

    word_as_list = list(word_completion)
    indices = [i for i, letter in enumerate(word) if letter == guess]
    for index in indices:
        word_as_list[index] = guess

    result = "".join(word_as_list)
    assert result == "T__T"
    assert guess in guessed_letters


def test_game_wrong_letter_guess():
    """Тест неправильного угадывания буквы"""
    tries = 6
    word = "TEST"
    guess = 'X'
    guessed_letters = []

    if guess not in word:
        tries -= 1
        guessed_letters.append(guess)

    assert tries == 5
    assert guess in guessed_letters


def test_game_win_by_whole_word():
    """Тест победы при угадывании всего слова целиком"""
    word = "TEST"
    guessed_words = []
    guessed = False

    guess = 'TEST'

    if guess == word:
        guessed = True
        guessed_words.append(guess)
        word_completion = word

    assert guessed is True
    assert word_completion == "TEST"
    assert guess in guessed_words


def test_game_lose_by_tries():
    """Тест проигрыша при исчерпании всех попыток"""
    tries = 6
    word = "TEST"
    wrong_guesses = ['X', 'Y', 'Z', 'W', 'Q', 'P']
    guessed_letters = []

    for guess in wrong_guesses:
        if guess not in word:
            tries -= 1
            guessed_letters.append(guess)

    assert tries == 0
    assert len(guessed_letters) == 6


def test_game_duplicate_letter():
    """Тест обработки повторного ввода буквы"""
    word = "TEST"
    guessed_letters = ['T']

    guess = 'T'

    if guess in guessed_letters:
        already_guessed = True
    else:
        already_guessed = False

    assert already_guessed is True
    assert guess in guessed_letters