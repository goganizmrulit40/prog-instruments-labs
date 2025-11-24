import re

from patterns import PATTERNS

"""
Функции для валидации данных
"""


def validate_field(field_name: str, value: str) -> bool:
    """Валидация одного поля по регулярному выражению"""
    if field_name not in PATTERNS:
        return True
    pattern = PATTERNS[field_name]
    return bool(re.match(pattern, value))


def validate_row(row: dict) -> bool:
    """Валидация всей строки"""
    for field, value in row.items():
        if not validate_field(field, value):
            return False
    return True


def process_csv_data(lines: list) -> list:
    """Обрабатывает данные CSV и возвращает номера невалидных строк"""

    row_data = [
        'telephone',
        'height',
        'inn',
        'identifier',
        'occupation',
        'latitude',
        'blood_type',
        'issn',
        'uuid',
        'date'
    ]
    patterns_list = [PATTERNS[field] for field in column_order]

    matrix = []
    for i, line in enumerate(lines[1:], 0):
        if not line.strip():
            continue

        fields = line.strip().split(';')

        if len(fields) < 10:
            continue

        else:
            clean_fields = [field.strip('"') for field in fields[:10]]
            matrix.append(clean_fields)

    invalid_rows = []
    for i in range(len(matrix)):
        if not validate_row(row_data):
            invalid_rows.append(i)

    return invalid_rows

