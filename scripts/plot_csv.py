#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os, glob

from IPython.display import display

from siesta_mlp import *

### FUNCTIONS ###

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
    ax.plot(data[:,ACCELERATION])
    #ax.plot(vel_computed)
    ax.axhline(y=0, color='k')
    ax.set_xlabel("t_i (400Hz)")
    ax.set_ylabel("Measure")
    ax.legend(["setpoint","postition","velocity","current","acceleration"])
    plt.title("Motor data reading (400Hz)")
    plt.show()

def add_acceleration_column(df):
    a = np.empty((df.shape[0],1,))
    a[:] = np.nan
    df.insert(3, "acceleration [rad/s^2]", a)

    display(df)
    df.to_csv('../data/2023-02-16--10-16-30_dataset_acc.csv',index=False, header=False)


### SCRIPT ###
list_of_files = glob.glob('../data/*.csv')
path = max(list_of_files, key=os.path.getctime)

#path = '../data/2023-02-21--14-05-03_dataset.csv'

print("Reading data from " + path)
df = pd.read_csv(path)
df = compute_acceleration(df)
plot_df(df)

#display(df.head(40))
