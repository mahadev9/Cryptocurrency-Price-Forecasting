
from datetime import datetime
from torch.utils.data import Dataset

from utility import *

import torch
import numpy as np
import pandas as pd


class CryptocurrencyDataset(Dataset):
    def __init__(self, inputs, actuals):
        self.X = inputs
        self.y = actuals

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def count_parameters(net, all=True):
    return sum(p.numel() for p in net.parameters() if p.requires_grad or all)


def model_train(model, optim, criterion, num_epochs, train_data, device):

    hist = np.zeros(num_epochs)
    start_time = datetime.now()

    for t in range(num_epochs):
        losses = 0
        count = 0
        for batch_idx, (inputs, actuals) in enumerate(train_data):
            inputs = inputs.to(device)
            actuals = actuals.to(device)
            predictions = model(inputs)

            loss = criterion(predictions, actuals)
            losses += loss.item()*inputs.shape[0]
            count += inputs.shape[0]

            optim.zero_grad()
            loss.backward()
            optim.step()

        print("Epoch: {} MSE: {}".format(t+1, losses/count))
        hist[t] = losses/count

    training_time = datetime.now() - start_time
    print("Training time: {}".format(training_time))

    return hist


def model_predictions(model, data_loader, device, scaler=None):

    predictions = np.array([])
    actuals = np.array([])

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            predictions = np.append(predictions, outputs.detach().cpu().numpy())
            actuals = np.append(actuals, labels.detach().cpu().numpy())

    if scaler:
        predict_test = pd.DataFrame(scaler.inverse_transform(predictions.reshape(-1, 1)))
        original_test = pd.DataFrame(scaler.inverse_transform(actuals.reshape(-1, 1)))
    else:
        predict_test = pd.DataFrame(predictions.reshape(-1, 1))
        original_test = pd.DataFrame(actuals.reshape(-1, 1))

    plot_graph(original_test, predict_test)

    return predictions

