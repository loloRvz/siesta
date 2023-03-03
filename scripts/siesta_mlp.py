#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import math

from datetime import datetime
from derivative import FiniteDifference, SavitzkyGolay, Kalman, Spectral
from scipy import signal

import torch
from torch.nn.init import xavier_uniform_
from torch.nn import MSELoss
from torch.optim import SGD
from torch.nn import Module
from torch.nn import Softsign
from torch.nn import Linear
from torch import Tensor
from torch.utils.data import random_split, Subset, DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import mean_squared_error


# Data columns
TIME, SETPOINT, POSITION, VELOCITY, CURRENT, VELOCITY_COMP, ACCELERATION_COMP = range(7)

SAMPLING_FREQ = 400 # Hz
TIME_UNIT = 0.001   # ms
LOAD_INERTIAS = np.array([1, \
                          224.5440144e-6, \
                          548.4378187e-6, \
                          287.7428055e-6])    #kg*m^2


### CLASSES ###

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        self.path = path
        self.df = pd.read_csv(path)
        self.X = np.empty([1,1]).astype('float32')
        self.y = np.empty([1]).astype('float32')
        
    # preprocess data (compute velocities and accelerations)
    def preprocess(self):
        data = (self.df).to_numpy()

        # Compute position derivatives if necessary
        fd = SavitzkyGolay(left=3, right=0, order=1, iwindow=True)
        
        resave = False
        # Compute velocity from position 
        if np.sum(np.isnan(data[:,VELOCITY_COMP])) > 1 or resave:
            data[:,VELOCITY_COMP] = fd.d(data[:,POSITION],data[:,TIME]*TIME_UNIT)
            #data[:,VELOCITY_COMP] = signal.savgol_filter(data[:,POSITION], window_length=9, polyorder=2, deriv=1, delta=1/SAMPLING_FREQ)
            resave = True
        # Compute acceleration from velocity or velocity_comp
        if np.sum(np.isnan(data[:,ACCELERATION_COMP])) > 1 or resave:
            data[:,ACCELERATION_COMP] = fd.d(data[:,VELOCITY_COMP],data[:,TIME]*TIME_UNIT)
            #data[:,ACCELERATION_COMP] = signal.savgol_filter(data[:,POSITION],window_length=9, polyorder=2, deriv=2, delta=1/SAMPLING_FREQ)
            resave = True
        
        # Save dataframe to csv if velocity or acceleration computed
        if resave:
            print("Resaving dataframe to csv")
            self.df = pd.DataFrame(data, columns = self.df.columns.values, dtype=np.float32)
            self.df.to_csv(self.path, index=False)

    # plot dataset
    def plot_data(self):
        data = self.df.to_numpy()

        # Make data a bit more readable - ignore units for now
        data[:,SETPOINT] -= math.pi
        data[:,POSITION] -= math.pi
        data[:,CURRENT] /= 1000
        data[:,VELOCITY_COMP] /= 10
        data[:,ACCELERATION_COMP] /= 1000

        fig,ax=plt.subplots()
        #ax.plot(data[:,TIME],data[:,SETPOINT:ACCELERATION_COMP+1])
        ax.plot(data[:,TIME],data[:,SETPOINT])
        ax.plot(data[:,TIME],data[:,POSITION])
        ax.plot(data[:,TIME],data[:,VELOCITY_COMP])
        ax.plot(data[:,TIME],data[:,ACCELERATION_COMP])
        ax.plot(data[:,TIME],data[:,CURRENT])
        ax.axhline(y=0, color='k')
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Amplitude")
        ax.legend([ "Setpoint [rad]", \
                    "Posistion [rad]", \
                    "Derived Velocity [10rad/s]", \
                    "Derived Accleration [1000rad/s^2]", \
                    "Current [A]"])                        
        plt.title("Motor data reading @400Hz")

        """ 
        fig2,ax=plt.subplots()
        plt.scatter(data[:,ACCELERATION_COMP],data[:,CURRENT])
        ax.set_xlabel("Acceleration [1000rad/^2]")
        ax.set_ylabel("Current [A]")                      
        plt.title("Current vs Acceleration Relation")
        """

        plt.show()

    # prepare inputs and labels for learning process
    def prepare_data(self,hist_length):
        data = self.df.to_numpy()
        self.X = np.resize(self.X,(data.shape[0],hist_length))

        # Get position error (setpoint-position)
        position_error = data[:,SETPOINT] - data[:,POSITION]
        #Get position error history
        for i in range(hist_length):
            self.X[:,i] = np.roll(position_error, i)
            self.X[:i,hist_length-(i+1)] = np.nan
        self.X = self.X[hist_length-1:,:] #Cut out t<0

        # Compute torque depending on load inertia (declared in filename)
        load_id = int(os.path.basename(self.path)[20])
        #self.y = data[:,ACCELERATION_COMP] * LOAD_INERTIAS[load_id]
        self.y = data[:,VELOCITY_COMP]
        self.y = self.y[hist_length-1:] #Cut out t<0

    # get indexes for train and test rows
    def get_splits(self, n_test=0.1):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        print("Training size: ", train_size)
        print("Testing size: ", test_size)

        test_range = range(train_size, train_size + test_size)

        # calculate the split
        #train, test = random_split(self, [train_size, test_size])
        train = Subset(self, range(train_size))
        test = Subset(self, test_range)

        train_dl = DataLoader(train, batch_size=32, shuffle=True,
                    pin_memory=True)
        test_dl = DataLoader(test, batch_size=1024, shuffle=False,
                    pin_memory=True)
        return train_dl, test_dl, test_range

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs, dev, layerDim):
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
        self.hidden5 = Linear(int(layerDim/2), n_outputs).to(self.dev)
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
            print(epoch)
            meanLoss = 0
            steps = 0
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                inputs, targets = inputs.to(dev), targets.to(dev).unsqueeze(1).float()
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
                model_scripted.save("../models/"+dt_string +
                                    "/delta_"+str(epoch)+".pt")
            #if epoch>= 100: break
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
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse, actuals, predictions


### SCRIPT ###
def main():
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Using GPU!")
    else:
        dev = "cpu"
        print("Using CPU D:")

    # Open measured data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/*.csv')
    path = max(list_of_files, key=os.path.getctime)
    #path = '../data/2023-02-22--15-00-04_dataset.csv'
    print("Opening: ",path)

    pos_hist_len = 5

    # Prepare dataset
    dataset = CSVDataset(path)
    dataset.preprocess()
    #dataset.plot_data()
    dataset.prepare_data(hist_length=pos_hist_len)
    train_dl, test_dl, test_range = dataset.get_splits() # Get data loaders

    # Network training
    os.makedirs("../models/"+os.path.basename(path), exist_ok=True)

    model = MLP(pos_hist_len, 1, dev, 32)
    train_model(train_dl, model, dev, os.path.basename(path), lr=0.01)   # train the model
    mse, true_vals, est_vals = evaluate_model(test_dl, model) # evaluate the model
    print('MSE: %.3f, RMSE: %.3f' % (mse, np.sqrt(mse)))

    # Plot validation dataset to time
    fig,ax=plt.subplots()
    ax.plot(dataset.df.to_numpy()[test_range,TIME], \
        np.column_stack((test_dl.dataset[:][0][:,0], \
        np.squeeze(true_vals)/10, \
        np.squeeze(est_vals)/10)))
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.legend(["Position error [rad]","Derived velocity [10rad/s]","Predicted velocities [10rad/s]"])
    plt.title("Model Validation")
    plt.show()


if __name__ == "__main__":
    main()
