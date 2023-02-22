#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob

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


# Data columns
SETPOINT, POSITION, VELOCITY, CURRENT, PERIOD, ACCELERATION = range(6)


### CLASSES ###

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = pd.read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.to_numpy()[1:,SETPOINT:CURRENT]
        self.y = df.to_numpy()[1:,ACCELERATION]
        print(self.X)
        print(self.y)

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

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs, dev, layerDim):
        super(MLP, self).__init__()
        self.dev = dev
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, layerDim).to(self.dev)
        xavier_uniform_(self.hidden1.weight).to(self.dev)
        self.act1 = Softsign().to(self.dev)
        # second hidden layer
        self.hidden2 = Linear(layerDim, int(layerDim/2)).to(self.dev)
        xavier_uniform_(self.hidden2.weight).to(self.dev)
        self.act2 = Softsign().to(self.dev)
        # third hidden layer
        self.hidden3 = Linear(int(layerDim/2), int(layerDim/2)).to(self.dev)
        xavier_uniform_(self.hidden3.weight).to(self.dev)
        self.act3 = Softsign().to(self.dev)
        # fourth hidden layer
        self.hidden4 = Linear(int(layerDim/2), int(layerDim/2)).to(self.dev)
        xavier_uniform_(self.hidden4.weight).to(self.dev)
        self.act4 = Softsign().to(self.dev)
        # fifth hidden layer and output
        self.hidden5 = Linear(int(layerDim/2), 6).to(self.dev)
        xavier_uniform_(self.hidden5.weight).to(self.dev)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = X.to(self.dev)
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # fourth hidden layer
        X = self.hidden4(X)
        X = self.act4(X)
        # fifth hidden layer and output
        X = self.hidden5(X)
        return X


### FUNCTIONS ###

def compute_acceleration(df):
    # Transform to numpy array for computations
    data = df.to_numpy()
    # Compute acceleration from velocity difference and divide by period
    data[:,ACCELERATION] = np.append(np.nan, np.diff(data[:,VELOCITY])/ data[1:,PERIOD]*1000 ) 
    # Transform back to dataframe
    return pd.DataFrame(data, columns = df.columns.values)

# prepare the dataset
def prepare_data(path, dev):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True,
                          pin_memory=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False,
                         pin_memory=True)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model, dev, dt_string, lr):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr, momentum=0.9)
    writer = SummaryWriter()
    # enumerate epochs
    epoch = 0
    try:
        while True:
            meanLoss = 0
            steps = 0
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                inputs, targets = inputs.to(dev), targets.to(dev)
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = model(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
                meanLoss = meanLoss + loss
                steps = steps + 1

            meanLoss = meanLoss/steps
            writer.add_scalar('Loss/train', meanLoss, epoch)
            if epoch % 5000 == 0:
                model_scripted = torch.jit.script(model)
                model_scripted.save("./models/"+dt_string +
                                    "/delta_"+str(epoch)+".pt")
            epoch = epoch + 1
    except KeyboardInterrupt:
        print('interrupted!')

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy()
        actual = targets.cpu().numpy()
        actual = actual.reshape((len(actual), 6))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse

# make a class prediction for one row of data
def predict(row, model, dev):
    # convert row to data
    row = Tensor([row]).to(dev)
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().cpu().numpy()
    return yhat[0]


### SCRIPT ###
def main():
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Using GPU!")
    else:
        dev = "cpu"
        print("Using CPU D:")

    os.makedirs("../models/", exist_ok=True)

    # prepare the data
    list_of_files = glob.glob('../data/*.csv')
    path = max(list_of_files, key=os.path.getctime)
    #path = '../data/2023-02-21--14-05-03_dataset.csv'
    train_dl, test_dl = prepare_data(path, dev)
    print(len(train_dl.dataset), len(test_dl.dataset))
"""     # define the network
    model = MLP(15, dev, 512)
    # train the model
    train_model(train_dl, model, dev, dt_string, lr=0.01)
    # evaluate the model
    mse = evaluate_model(test_dl, model)
    print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))

    predictions = []
    gt = []
    for i in range(0, len(test_dl.dataset)):
        row = test_dl.dataset[i][0]
        predictions.append(predict(row, model, dev))
        gt.append(test_dl.dataset[i][1])

    # Rearrange lists
    predictions = [list(x) for x in zip(*predictions)]
    gt = [list(x) for x in zip(*gt)]
    plt.plot(predictions[0], color='red')
    plt.plot(gt[0], color='blue')
    plt.show() """

    # torch.save(
    #     model, '/home/eugenio/catkin_ws/src/delta_control/scripts/NNIdentification/fixed_delta.pt')

    # make a single prediction (expect class=1)
    # row = [0.00632, 18.00, 2.310, 0, 0.5380, 6.5750,
    #        65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98]
    # yhat = predict(row, model,dev)
    # print('Predicted: %.3f' % yhat)


if __name__ == "__main__":
    main()
