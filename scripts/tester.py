#! /usr/bin/env python3

import torch
from torch.nn.init import xavier_uniform_
from torch.nn import MSELoss
from torch.optim import SGD
from torch.nn import Module
from torch.nn import Softsign
from torch.nn import Linear
from torch import Tensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from numpy import sqrt
from numpy import vstack
import matplotlib.pyplot as plt
import numpy as np


# pytorch mlp for regression

# dataset definition


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[1:, :-6].astype('float32')
        self.y = df.values[1:, -6:].astype('float32')
        print(self.X.shape)
        print(self.y.shape)
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 6))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.1):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


def predict(row, model, dev):
    # convert row to data
    row = Tensor([row]).to(dev)
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().cpu().numpy()
    return yhat[0]


if torch.cuda.is_available():
    dev = "cuda:0"
    print("Using GPU!")
else:
    dev = "cpu"
    print("Using CPU D:")

# prepare the data
path = '/home/lolo/siesta_ws/src/siesta/data/2023-02-15--18-44-29_dataset.csv'
# load the dataset
dataset = CSVDataset(path)
# load the model
model = torch.jit.load(
    '/home/eugenio/catkin_ws/src/delta_control/scripts/NNIdentification/models/Nov-12-2022-09:04/delta_5000.pt')
model.eval()
model = model.to(dev)
# predict the dataset
predictions = []
gt = []
for i in range(0, len(dataset)):
    row = dataset[i][0]
    predictions.append(predict(row, model, dev))
    gt.append(dataset[i][1])

# Rearrange lists
predictions = [list(x) for x in zip(*predictions)]
gt = [list(x) for x in zip(*gt)]
rmse = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
for i in range(0, 6):
    rmse[i] = np.sqrt(np.mean((np.array(predictions[i])-np.array(gt[i]))**2))

print("RMSE ", rmse)
plt.plot(predictions[0], color='red')
plt.plot(gt[0], color='blue')
plt.show()
# yhat = predict(row, model, dev)
# print('Predicted: %.3f' % yhat)
