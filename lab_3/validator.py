import re

from patterns import PATTERNS

"""
Функции для валидации данных по регулярным выражениям
"""


def validate_field(row: int, matrix: list, column: int, pattern: str) -> bool:
    """
    Валидация одного значения таблицы по регулярному выражению
    :param row: индекс строки в матрице
    :param matrix: сама матрица (список списков)
    :param column: индекс столбца в матрице
    :param pattern: регулярное выражения для валидации
    :return: 1, если значение соот-ет паттерну, 0 - иначе
    """
    value = str(matrix[row][column]).strip()
    return bool(re.match(pattern, value))


def validate_row(matrix: list, patterns: list) -> list[int]:
    """
    Валидация всей строки: находит индексы строк, которые содержат невалидные данные.
    Проверяет значения столбцов, если обнаружено несоот-ие паттерну,
    добавляет индекс в невалидные строки.
    :param matrix: матрица для валидации
    :param patterns: список регулярных выражений
    :return: список индексов строк, где находятся невалидные данные
    """
    invalid_rows = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if not validate_field(i, matrix, j, patterns[j]):
                invalid_rows.append(i)
                break
    return invalid_rows


def process_csv_data(matrix: list) -> list:
    """
    Обрабатывает данные CSV и возвращает номера невалидных строк
    :param matrix: матрица для валидации
    :return: список индексов строк, где находятся невалидные данные
    """
    return validate_row(matrix, PATTERNS)
