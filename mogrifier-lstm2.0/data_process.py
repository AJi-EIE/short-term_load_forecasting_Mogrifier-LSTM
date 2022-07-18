# -*- coding:utf-8 -*-

import os
import random

import numpy as np
import pandas as pd
from regex import D
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data():
    """
    :return: normalized dataframe
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/data.csv'
    df = pd.read_csv(path, encoding='gbk')
    columns = df.columns
    df.fillna(method = 'bfill')
    MAX = np.max(df[columns[1]])
    MIN = np.min(df[columns[1]])
    df[columns[1]] = (df[columns[1]] - MIN) / (MAX - MIN)

    return df, MAX, MIN


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


# Multivariate-MultiStep-LSTM data processing.
def nn_seq_mm(B, num):
    print('data processing...')
    data, m, n = load_data()
    load = data[data.columns[1]]
    load = load.tolist()
    data = data.values.tolist()
    seq = []
    window = 96

    for i in range(0, len(data) - window - num, num):
        train_seq = []
        train_label = []

        for j in range(i, i + window):
            x = [load[j]]
            for c in range(2, 8):
                x.append(data[j][c])
            train_seq.append(x)

        for j in range(i + window, i + window + num):
            train_label.append(load[j])

        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label))

    # print(seq[-1])
    Dtr = seq[0:int(len(seq) * 0.7)]
    Dte = seq[int(len(seq) * 0.7):len(seq)]

    train_len = int(len(Dtr) / B) * B
    test_len = int(len(Dte) / B) * B
    Dtr, Dte = Dtr[:train_len], Dte[:test_len]

    train = MyDataset(Dtr)
    test = MyDataset(Dte)
    Dtr = DataLoader(dataset=train, batch_size=B, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=B, shuffle=False, num_workers=0)

    return Dtr, Dte, m, n


# Multivariate-SingleStep-LSTM data processing.
def nn_seq_ms(B):
    print('data processing...')
    data, m, n = load_data()
    load = data[data.columns[1]]
    load = load.tolist()
    data = data.values.tolist()
    seq = []

    window = 96
    for i in range(len(data) - window):
        train_seq = []
        train_label = []
        for j in range(i, i + window):
            x = [load[j]]
            for c in range(2, 8):
                x.append(data[j][c])
            train_seq.append(x)
        train_label.append(load[i + window])
        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label))

    Dtr = seq[0:int(len(seq) * 0.7)]
    Dte = seq[int(len(seq) * 0.7):len(seq)]

    train_len = int(len(Dtr) / B) * B
    test_len = int(len(Dte) / B) * B
    Dtr, Dte = Dtr[:train_len], Dte[:test_len]

    train = MyDataset(Dtr)
    test = MyDataset(Dte)

    Dtr = DataLoader(dataset=train, batch_size=B, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=B, shuffle=False, num_workers=0)

    return Dtr, Dte, m, n


# Univariate-SingleStep-LSTM data processing.
def nn_seq(B):
    print('data processing...')
    data, m, n = load_data()
    load = data[data.columns[1]]
    load = load.tolist()
    load = torch.FloatTensor(load).view(-1)
    data = data.values.tolist()
    seq = []
    window = 96

    for i in range(len(data) - window):
        train_seq = []
        train_label = []
        for j in range(i, i + window):
            train_seq.append(load[j])
        # for c in range(2, 8):
        #     train_seq.append(data[i + 24][c])
        train_label.append(load[i + window])
        train_seq = torch.FloatTensor(train_seq).view(-1)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label))
    # print(seq[:5])

    Dtr = seq[0:int(len(seq) * 0.7)]
    Dte = seq[int(len(seq) * 0.7):len(seq)]

    train_len = int(len(Dtr) / B) * B
    test_len = int(len(Dte) / B) * B
    Dtr, Dte = Dtr[:train_len], Dte[:test_len]

    train = MyDataset(Dtr)
    test = MyDataset(Dte)

    Dtr = DataLoader(dataset=train, batch_size=B, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=B, shuffle=False, num_workers=0)

    return Dtr, Dte, m, n


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x)) * 100

def get_rmse(x, y):

    return np.sqrt(np.mean(np.square(x - y)))

def get_mse(x, y):

    return np.mean(np.square(x - y))

def get_mae(x, y):

    return np.mean(np.abs(x - y))
