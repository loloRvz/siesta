#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from enum import Enum
from IPython.display import display

from mlp import CSVDataset

SETPOINT, POSITION, VELOCITY, ACCELERATION, CURRENT, PERIOD = range(6)

### FUNCTIONS ###

def compute_acceleration(df):
    # Transform to numpy array for computations
    data = df.to_numpy()

    # Compute acceleration from velocity difference and divide by period
    data[:,ACCELERATION] = np.append(np.nan, np.diff(data[:,VELOCITY])/ data[1:,PERIOD]*1000 ) 

    # Transform back to dataframe
    return pd.DataFrame(data, columns = df.columns.values)

def plot_df(df):
    data = df.to_numpy()

    # Make data a bit more readable - ignore units for now
    data[:,SETPOINT] -= math.pi
    data[:,POSITION] -= math.pi
    data[:,CURRENT] /= 1000
    data[:,ACCELERATION] /= 100
    data[:,VELOCITY] /= 5

    #Compute velocity from position values
    #vel_computed = np.append(np.nan, np.diff(data[:,POSITION])/ data[1:,PERIOD]*1000 ) / 10

    fig,ax=plt.subplots()
    ax.plot(data[:,SETPOINT:CURRENT+1])
    #ax.plot(vel_computed)
    ax.axhline(y=0, color='k')
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

#path = '../data/2023-02-16--10-16-30_dataset.csv'
path = '../data/2023-02-20--14-58-08_dataset.csv'

df = pd.read_csv(path)
df = compute_acceleration(df)
plot_df(df)


display(df.head(40))
