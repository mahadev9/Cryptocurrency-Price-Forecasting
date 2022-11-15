
from datetime import datetime
from torch.utils.data import Dataset

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
        for batch_idx, (inputs, actuals) in enumerate(train_data):
            inputs = inputs.to(device)
            actuals = actuals.to(device)
            predictions = model(inputs)

            loss = criterion(predictions, actuals)
            if ((t+1) % 10) == 0:
                print("Epoch: {} MSE: {}".format(t+1, loss.item()))
            hist[t] = loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

    training_time = datetime.now() - start_time
    print("Training time: {}".format(training_time))

    return predictions, hist


def model_predictions(model, inputs, actuals, device):

    # make predictions
    with torch.no_grad():
        inputs = inputs.to(device)
        predictions = model(inputs)

    predict_test = pd.DataFrame(predictions.detach().cpu().numpy())
    original_test = pd.DataFrame(actuals.detach().cpu().numpy())

    sns.set_style("darkgrid")

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 1, 1)
    ax = sns.lineplot(x=original_test.index,
                      y=original_test[0], label="Data", color='royalblue')
    ax = sns.lineplot(x=predict_test.index,
                      y=predict_test[0], label="Testing Prediction", color='tomato')
    ax.set_title('Stock price', size=14, fontweight='bold')
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Cost (USDT)", size=14)
    ax.set_xticklabels('', size=10)

    fig.set_figheight(6)
    fig.set_figwidth(16)

    return predictions


def mean_squared_error(actuals, predictions):
    return np.mean(np.power(actuals - predictions, 2))


def rmse(actuals, predictions, train_test):
    actuals = actuals.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    trainScore = np.sqrt(mean_squared_error(actuals, predictions))
    print('{} Score: {.2f} RMSE'.format(train_test, trainScore))
