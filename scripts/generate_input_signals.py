#! /usr/bin/env python3

import csv
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## INPUT SIGNAL PARAMETERS ##
# General
DATA_LENGTH = 120   # [s]
CTRL_FREQ = 200     # [Hz]
READ_FREQ = 400     # [Hz]

# Step input
STEP_FREQ = 5       # [Hz]
STEP_VAR = 0.3      # [Rad]

# Chirp input 
CHRP_AMPL = 0.3               # [Rad]
CHRP_FREQ1 = 0.05 *2*math.pi   # [Rad/s]
CHRP_FREQ2 =   40 *2*math.pi   # [Rad/s]
CHRP_PERIOD = 5               # [s]

# Flight data input
FLIT_FILE = "23-02-08--21-26-11_ID5.csv"

# Mixed input containing all types of data input
MIXD_INTERVAL = 5  # [s]

# Enum
STEP_IDX, CHRP_IDX, FLIT_IDX, MIXD_IDX = range(4)



## FUNCTIONS ##
def chirp_signal(time):
    time = time % CHRP_PERIOD
    return CHRP_AMPL*math.cos(CHRP_FREQ1*time+(CHRP_FREQ2-CHRP_FREQ1)*time*time/(2*CHRP_PERIOD))


### SCRIPT ###
def main():
    # Get global path of this script
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Compute random step input
    step_inputs = np.random.normal(0, STEP_VAR, DATA_LENGTH*STEP_FREQ)
    step_inputs = np.repeat(step_inputs, CTRL_FREQ/STEP_FREQ)

    # Compute chirp signal input
    chrp_inputs = np.array([chirp_signal(t) for t in np.arange(0,DATA_LENGTH,1/CTRL_FREQ)])

    # Parse flight data setpoints
    flight_data_path = dir_path + "/../data/flight_data/23-02-08--21-26-11_ID5.csv"
    flight_data_df = pd.read_csv(flight_data_path)
    flight_data_np = flight_data_df["setpoint[rad]"].to_numpy()[10000:] # remove first 10'000 data points (avoid constant input)
    flight_data_df = pd.DataFrame(flight_data_np, columns = ['flit'])

    # Combine inputs into dataframe
    df = pd.DataFrame(np.column_stack((step_inputs,chrp_inputs)), columns=['step','chrp'])
    df["flit"] = flight_data_df

    # Compute mixed types input
    inputs_array = df.to_numpy()
    mixd_np = np.zeros(inputs_array.shape[0])

    n = 0
    seg_size = MIXD_INTERVAL*CTRL_FREQ
    arr = np.zeros((seg_size,3))
    while n < DATA_LENGTH*CTRL_FREQ:
        # This works for some reason, don't touch
        for i in range(3):
            if n+i*seg_size < DATA_LENGTH*CTRL_FREQ:
                arr[:,i] = np.resize( inputs_array[n+i*seg_size:n+(i+1)*seg_size,i] , arr[:,i].shape )
        mixd_np[n:n+3*seg_size] = np.resize( np.transpose(arr).reshape((1,-1)), mixd_np[n:n+3*seg_size].shape )
        n = n + 3*seg_size

    df["mixd"] = pd.DataFrame(mixd_np, columns =  ['mixd'])

    print("Computed inputs:")
    print(df)
    df.plot()
    #plt.show()

    # Write dataframe to csv file
    df.to_csv(dir_path + "/../data/input_signals/signals.csv", index=False)




if __name__ == "__main__":
    main()

