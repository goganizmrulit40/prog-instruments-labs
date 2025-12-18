import os
import numpy as np
import torch
from faker import Faker
import random
from typing import Tuple, Optional, List, Dict, Any

from torch.nn import init
from tqdm import tqdm
from babel.dates import format_date
from torchtext.data import Field, BucketIterator, TabularDataset
import pandas as pd
from sklearn.model_selection import train_test_split
fake = Faker()
Faker.seed(12345)
random.seed(12345)

# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

# change this if you want it to work with another language
LOCALES = ['en_US']


def load_date() -> Tuple[Optional[str], Optional[str], Optional[object]]:
    """Генерация одной даты в человекочитаемом и машиночитаемом формате."""
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS), locale='en_US')
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt


def load_dataset(m: int, max_attempts_multiplier: int = 2) -> List[List[str]]:
    """Загрузка датасета с m примерами."""
    dataset = []
    max_attempts = m * max_attempts_multiplier

    for attempt in range(max_attempts):
        if len(dataset) >= m:
            break

        human_readable, machine_readable, _ = load_date()
        if human_readable is not None:
            dataset.append([human_readable, machine_readable])

    if len(dataset) < m:
        print(f"Предупреждение: сгенерировано только {len(dataset)} из {m} примеров")

    return dataset


def prepare_data(dataset_path: str = r"../dataset/date-normalization",
                 dataset_size: int = 10,
                 debug: bool = False) -> Tuple[str, str]:
    """Подготовка тренировочных и валидационных данных."""
    if debug:
        dataset_size = 10
        train_file = os.path.join(dataset_path, "train_small.csv")
        eval_file = os.path.join(dataset_path, "eval_small.csv")
    else:
        train_file = os.path.join(dataset_path, "train.csv")
        eval_file = os.path.join(dataset_path, "eval.csv")
    if not os.path.exists(train_file) and not os.path.exists(train_file):
        dataset = load_dataset(dataset_size)
        source, target = zip(*dataset)
        X_train, X_test, y_train, y_test = train_test_split(source, target, random_state=42, test_size=0.2)
        train_df = pd.DataFrame()
        train_df["source"], train_df["target"] = X_train, y_train
        eval_df = pd.DataFrame()
        eval_df["source"], eval_df["target"] = X_test, y_test
        train_df.to_csv(train_file, index=False)
        eval_df.to_csv(eval_file, index=False)
    return train_file, eval_file


def dataset2dataloader(dataset_path: str,
                       batch_size: int = 10,
                       dataset_size: int = 10,
                       debug: bool = False) -> Tuple[BucketIterator, BucketIterator, object, object]:
    """Конвертация датасета в DataLoader для PyTorch."""
    train_csv, dev_csv = prepare_data(dataset_path, dataset_size=dataset_size, debug=debug)

    def tokenizer(text):
        return list(text)

    # Определение формата данных
    SOURCE = Field(sequential=True, tokenize=tokenizer, lower=False)
    TARGET = Field(sequential=True, tokenize=tokenizer, lower=False, init_token="<start>", eos_token="<end>")
    train, val = TabularDataset.splits(
        path='', train=train_csv, validation=dev_csv, format='csv', skip_header=True,
        fields=[('source', SOURCE), ('target', TARGET)])

    SOURCE.build_vocab(train)
    TARGET.build_vocab(train)

    train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.sent), shuffle=False)
    val_iter = BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.sent), shuffle=False)

    # 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
    # test_iter = data.Iterator(dataset=test, batch_size=128, train=False, sort=False, device=DEVICE)

    return train_iter, val_iter, SOURCE.vocab, TARGET.vocab