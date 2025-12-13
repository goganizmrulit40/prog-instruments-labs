from keras.utils import to_categorical

from dataloader import load_dataset, dataset2dataloader
from models import SimpleNMT
from torch import optim
import torch.nn as nn
import torch
import numpy as np
from pprint import pprint
from tqdm import tqdm

CONFIG = {
    'epoch': 500,
    'learning_rate': 0.001,
    'hidden_size': 64,
    'batch_size': 10,
    'Tx': 25,
    'Ty': 10
}


def load_data(CONFIG, debug=True):
    """Загрузка и подготовка данных"""
    train_iter, val_iter, source_vocab, target_vocab = dataset2dataloader(
        dataset_path=r"../dataset/date-normalization",
        batch_size=CONFIG['batch_size'],
        dataset_size=10000,
        debug=debug
    )
    return train_iter, val_iter, source_vocab, target_vocab


def create_model(source_vocab_size, target_vocab_size, CONFIG):
    """Создание модели NMT"""
    model = SimpleNMT(
        in_vocab_size=source_vocab_size,
        out_vocab_size=target_vocab_size,
        in_hidden_size=CONFIG['hidden_size'],
        out_hidden_size=CONFIG['hidden_size'],
        output_size=target_vocab_size,
        with_attention=True
    )
    return model


def create_optimizer_and_criterion(model, CONFIG):
    """Создание оптимизатора и функции потерь"""
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


def create_embedding_layers(source_vocab_size, target_vocab_size):
    """Создание embedding слоев с one-hot кодированием"""
    embed_layer1 = nn.Embedding(
        source_vocab_size, source_vocab_size,
        _weight=torch.from_numpy(np.eye(source_vocab_size))
    )
    embed_layer2 = nn.Embedding(
        target_vocab_size, target_vocab_size,
        _weight=torch.from_numpy(np.eye(target_vocab_size))
    )
    return embed_layer1, embed_layer2


def prepare_batch(batch, embed_layer1, embed_layer2):
    """Подготовка данных батча"""
    Xin = embed_layer1(batch.source.t().long()).float()
    Yin = embed_layer2(batch.target.t()[:, :-1].long()).float()
    Yout = batch.target.t()[:, 1:]
    return Xin, Yin, Yout


def create_initial_hidden(batch_size, hidden_size):
    """Создание начального скрытого состояния"""
    return torch.zeros(1, batch_size, hidden_size)


def train_step(model, batch, optimizer, criterion, embed_layers, hidden_size):
    """Один шаг обучения"""
    optimizer.zero_grad()

    Xin, Yin, Yout = prepare_batch(batch, *embed_layers)
    init_hidden = create_initial_hidden(len(Xin), hidden_size)

    logits = model(Xin, init_hidden, Yin)
    loss = criterion(logits.view(-1, logits.shape[-1]), Yout.flatten())

    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(model, train_iter, optimizer, criterion, embed_layers, CONFIG):
    """Обучение модели на всех эпохах"""
    model.train()

    for ep in range(CONFIG['epoch']):
        epoch_loss = 0

        for batch in train_iter:
            loss_value = train_step(model, batch, optimizer, criterion, embed_layers, CONFIG['hidden_size'])
            epoch_loss += loss_value

        if ep % (CONFIG['epoch'] // 10) == 0:
            print(f"Эпоха {ep}, loss: {epoch_loss}")


if __name__ == "__main__":
    # 1. Загрузка данных
    train_iter, val_iter, source_vocab, target_vocab = load_data(CONFIG)
    source_vocab_size = len(source_vocab.stoi)
    target_vocab_size = len(target_vocab.stoi)

    # 2. Создание модели
    model = create_model(source_vocab_size, target_vocab_size, CONFIG)

    # 3. Создание оптимизатора
    optimizer, criterion = create_optimizer_and_criterion(model, CONFIG)

    # 4. Создание embedding слоев
    embed_layers = create_embedding_layers(source_vocab_size, target_vocab_size)

    # 5. Обучение
    train_model(model, train_iter, optimizer, criterion, embed_layers, CONFIG)

    """ 不使用 attention
    dataset_size : 10000
    loss 940.5139790773392
    loss 151.68325132876635
    loss 17.91189043689519
    loss 8.461621267197188
    loss 0.4571912245155545
    loss 4.067497536438168
    loss 0.02432645454427984
    loss 0.022933890589229122
    loss 1.740354736426525
    loss 2.7019595313686295
    monday may 7 1983 --> 1983-05-07
    19 march 1998 --> 1998-03-19
    18 jul 2008 --> 2008-07-18
    9/10/70 --> 1970-09-10
    thursday january 1 1981 --> 1981-01-01
    thursday january 26 2015 --> 2015-01-26
    saturday april 18 1990 --> 1990-04-18
    sunday may 12 1988 --> 1988-05-12
    """

    """使用attention
    loss 870.4544065594673
    loss 65.41884177550673
    loss 53.339022306521656
    loss 0.08635593753569992
    loss 0.057157438381182146
    loss 0.0006471980702968949
    loss 0.09261544834953384
    loss 0.000922315769471993
    loss 0.00961817828419953
    loss 0.06814217135979561
    monday may 7 1983 --> 1983-05-07
    19 march 1998 --> 1998-03-19
    18 jul 2008 --> 2008-07-18
    9/10/70 --> 1970-09-10
    thursday january 1 1981 --> 1981-01-01
    thursday january 26 2015 --> 2015-01-26
    saturday april 18 1990 --> 1990-04-18
    sunday may 12 1988 --> 1988-05-12
    """
