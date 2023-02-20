#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from IPython.display import display

from mlp import CSVDataset

SETPOINT, POSITION, VELOCITY, ACCELERATION, CURRENT, PERIOD = range(6)

### FUNCTIONS ###

def compute_acceleration(df):
    # Transform to numpy array for computations
    data = df.to_numpy()
    # Compute acceleration from velocity difference and divide by period
    data[:,ACCELERATION] = np.append(np.nan, np.diff(data[:,VELOCITY])) / data[:,PERIOD] * 1000
    # Transform back to dataframe
    df = pd.DataFrame(data, columns = df.columns.values)

def plot_df(df):
    data = df.to_numpy()
    data[:,CURRENT] /= 1000
    data[:,ACCELERATION] /= 100
    fig,ax=plt.subplots()
    ax2=ax.twinx()

    #Compute velocity
    vel_computed = np.append(np.nan, np.diff(data[:,POSITION])) / data[:,PERIOD] * 1000

    ax.plot(data[:,SETPOINT:CURRENT+1])
    ax.plot(vel_computed)
    ax.set_xlabel("time")
    ax.set_ylabel("tick")
    ax.legend(df.columns.values)
    plt.show()

def add_acceleration_column(df):
    a = np.empty((df.shape[0],1,))
    a[:] = np.nan
    df.insert(3, "acceleration [rad/s^2]", a)

    display(df)
    df.to_csv('../data/2023-02-16--10-16-30_dataset_acc.csv',index=False, header=False)


### SCRIPT ###

path = '../data/2023-02-15--18-44-29_dataset.csv'
df = pd.read_csv(path)
compute_acceleration(df)
display(df)
plot_df(df)
