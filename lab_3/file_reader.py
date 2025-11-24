import chardet
import pandas as pd

"""
Чтение csv-файла с проверкой кодировки
"""


def read_csv_file(file_path: str) -> list:
    """
    Читает CSV файл с проверкой кодировки
    С помощью pandas считывает данные и преобразует в список списков
    :param file_path: путь к csv файлу
    :return: матрица данных - список списков
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)

    encoding = result['encoding']
    print(f"Кодировка файла: {encoding}")

    matrix = pd.read_csv(file_path, sep=';', encoding=encoding)
    return matrix.values.tolist()
