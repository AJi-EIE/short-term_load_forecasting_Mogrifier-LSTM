# -*- coding:utf-8 -*-

from itertools import chain

import torch
from scipy.interpolate import make_interp_spline
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import LSTM, BiLSTM, MogLSTM, CNNLSTMModel
from data_process import nn_seq, nn_seq_ms, nn_seq_mm, device, get_mape, get_rmse , get_mse, get_mae, setup_seed

from log import get_log

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

setup_seed(20)

logger = get_log('./log/log_Moglstm_96_1.txt')




def train(conf, path, flag):
    input_size, hidden_size, num_layers, batch_size, mogrify_steps  = conf["input_size"], conf["hidden_size"], conf["num_layers"], conf["lstm_batch_size"], conf["mogrify_steps"]
    output_size = conf["output_size"]

    model_input_size = conf["model_input_size"]

    if flag == 'us':
        Dtr, Dte, m, n = nn_seq(B=batch_size)
    elif flag == 'ms':
        Dtr, Dte, m, n = nn_seq_ms(B=batch_size)
    else:
        Dtr, Dte, m, n = nn_seq_mm(B=batch_size, num=output_size)

    if conf["model_name"] == "BiLSTM":
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
    elif conf["model_name"] == "LSTM":
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
    elif conf["model_name"] == "MogLSTM":
        model = MogLSTM(model_input_size, hidden_size, mogrify_steps, output_size, batch_size).to(device)
    else:
        model = CNNLSTMModel(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)


    loss_function = nn.MSELoss().to(device)

    if conf["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], momentum=conf["momentum"], weight_decay=conf["weight_decay"])


    # training
    loss = 0
    for i in range(conf["local_epochs"]):
        cnt = 0
        for (seq, label) in Dtr:
            cnt += 1
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if cnt % 100 == 0:
            #     print('epoch', i, ':', cnt - 100, '~', cnt, loss.item())
    

        logger.info("Epoch [%d], Loss: %.12f" % (i + 1, loss.item()))
        

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, path)


def test(conf, path, flag):

    input_size, hidden_size, num_layers, batch_size, mogrify_steps  = conf["input_size"], conf["hidden_size"], conf["num_layers"], conf["lstm_batch_size"], conf["mogrify_steps"]
    output_size = conf["output_size"]

    model_input_size = conf["model_input_size"]

    if flag == 'us':
        Dtr, Dte, m, n = nn_seq(B=batch_size)
    elif flag == 'ms':
        Dtr, Dte, m, n = nn_seq_ms(B=batch_size)
    else:
        Dtr, Dte, m, n = nn_seq_mm(B=batch_size, num=output_size)

    pred = []
    y = []

    print('loading model...')
    if conf["model_name"] == "BiLSTM":
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
    elif conf["model_name"] == "LSTM":
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
    elif conf["model_name"] == "MogLSTM":
        model = MogLSTM(model_input_size, hidden_size, mogrify_steps, output_size, batch_size).to(device)
    else:
        model = CNNLSTMModel(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)


    model.load_state_dict(torch.load(path)['model'])
    model.eval()

    print('predicting...')
    for (seq, target) in Dte:
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    
    dataframe = pd.DataFrame({'y':y,'pred':pred})

    dataframe.to_csv(r"./test_csv/MogLSTM_96_1.csv", index= False)
    # eval
    mse = get_mse(y, pred)
    rmse = get_rmse(y, pred)
    mae = get_mae(y, pred)
    mape = get_mape(y, pred)

    logger.info("MSE: %.12f" % (mse))
    logger.info("RMSE: %.12f" % (rmse))
    logger.info("MAE: %.12f" % (mae))
    logger.info("MAPE: %.12f" % (mape))

    # plot
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.switch_backend('agg')
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.savefig("./pic_result/MogLSTM_96_1.jpg")
