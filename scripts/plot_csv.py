#! /usr/bin/env python3

import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt

from mlp import CSVDataset



### FUNCTIONS ###

# prepare the dataset
def fetch_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    print(dataset)

def plot_data():
    print("here")

def add_acceleration_column():
    df = pd.read_csv(path, header=None)
    a = np.empty((df.shape[0]-1,1,))
    a[:] = np.nan
    a = np.append("acceleration [rad/s^2]",a)
    df.insert(3, "3", a)
    display(df)
    df.to_csv('../data/2023-02-16--10-16-30_dataset_acc.csv',index=False, header=False)


### SCRIPT ###

# prepare the data
path = '../data/2023-02-16--10-16-30_dataset.csv'
#fetch_data(path)

