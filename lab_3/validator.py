import re

from patterns import PATTERNS

"""
Функции для валидации данных
"""


def validate_field(row: int, matrix: list, column: int, pattern: str) -> bool:
    """Валидация одного поля по регулярному выражению"""
    value = str(matrix[row][column]).strip()
    return bool(re.match(pattern, value))


def validate_row(matrix: list, patterns: list) -> list[int]:
    """Валидация всей строки"""
    invalid_rows = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if not validate_field(i, matrix, j, patterns[j]):
                invalid_rows.append(i)
                break
    return invalid_rows


def process_csv_data(matrix: list) -> list:
    """Обрабатывает данные CSV и возвращает номера невалидных строк"""
    return validate_row(matrix, PATTERNS)
