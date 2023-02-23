#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import math

from derivative import dxdt, FiniteDifference, SavitzkyGolay, Spectral, TrendFiltered

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
TIME, SETPOINT, POSITION, VELOCITY, CURRENT, VELOCITY_COMP, ACCELERATION_COMP = range(7)

TIME_UNIT = 0.001   # ms


### CLASSES ###

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        self.df = pd.read_csv(path, header=None)
        # store the inputs and outputs
        #self.X = df.to_numpy()[1:,SETPOINT:CURRENT]

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

# prepare dataset
def prepare_data(path):
    # Load dataset & convert to np
    df= pd.read_csv(path)
    data = df.to_numpy()

    fd = SavitzkyGolay(left=.005, right=.005, order=2, periodic=True)

    # Compute velocity from position 
    if True: #np.sum(np.isnan(data[:,VELOCITY_COMP])) > 1:
        data[:,VELOCITY_COMP] = fd.d(data[:,POSITION],data[:,TIME]*TIME_UNIT)
        resave = True

    # Compute acceleration from velocity or velocity_comp
    if True: #np.sum(np.isnan(data[:,ACCELERATION_COMP])) > 1:
        data[:,ACCELERATION_COMP] = fd.d(data[:,VELOCITY_COMP],data[:,TIME]*TIME_UNIT)
        resave = True
    
    # Save dataframe to csv if velocity or acceleration computed
    if resave:
        print("Resaving dataframe to csv")
        df = pd.DataFrame(data, columns = df.columns.values)
        df.to_csv(path, index=False)

    return df

# plot dataset
def plot_df(df):
    data = df.to_numpy()

    # Make data a bit more readable - ignore units for now
    data[:,SETPOINT] -= math.pi
    data[:,POSITION] -= math.pi
    data[:,CURRENT] /= 1000
    data[:,VELOCITY] /= 10
    data[:,VELOCITY_COMP] /= 10
    data[:,ACCELERATION_COMP] /= 1000

    fig,ax=plt.subplots()
    #ax.plot(data[:,TIME],data[:,SETPOINT:ACCELERATION_COMP+1])
    ax.plot(data[:,TIME],data[:,SETPOINT])
    ax.plot(data[:,TIME],data[:,POSITION])
    ax.plot(data[:,TIME],data[:,VELOCITY])
    ax.plot(data[:,TIME],data[:,VELOCITY_COMP])
    ax.plot(data[:,TIME],data[:,ACCELERATION_COMP])
    ax.plot(data[:,TIME],data[:,CURRENT])
    ax.axhline(y=0, color='k')
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.legend([ "Setpoint [rad]", \
                "Posistion [rad]", \
                "Velocity [10rad/s]", \
                "Derived Velocity [10rad/s]", \
                "Derived Accleration [1000rad/s^2]", \
                "Current [A]"])
    plt.title("Motor data reading @400Hz")
    plt.show()

# train model
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

# evaluate model
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

# make class prediction for one row of data
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
    # prepare the data
    list_of_files = glob.glob('../data/*.csv')
    path = max(list_of_files, key=os.path.getctime)
    #path = '../data/2023-02-22--15-00-04_dataset.csv'
    print("Opening: ",path)

    df = prepare_data(path)
    plot_df(df)



if __name__ == "__main__":
    main()
