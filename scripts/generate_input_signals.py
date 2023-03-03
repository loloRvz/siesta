#! /usr/bin/env python3

import csv
import os
import math
import numpy as np
import pandas as pd


## INPUT SIGNAL PARAMETERS ##
# General
DATA_LENGTH = 120   # [s]
CTRL_FREQ = 200     # [Hz]
READ_FREQ = 400     # [Hz]

# Step input
STEP_FREQ = 5       # [Hz]
STEP_VAR = 0.1      # [Rad]

# Chirp input 
CHRP_AMPL = 0.5     # [Rad]
CHRP_FREQ1 = 10     # [Hz]
CHRP_FREQ2 = 40     # [Hz]
CHRP_PERIOD = 2     # [s]

# Flight data input
FLIT_FILE = "23-02-08--21-26-11_ID5.csv"


## FUNCTIONS ##
def chirp_signal(time):
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
    flight_data_path = dir_path + "/../data/flight_data/23-02-08--21-26-11_ID5"
    flight_data = pd.read_csv(flight_data_path)
    print(flight_data.head())


    # Combine inputs into dataframe
    df = pd.DataFrame(np.column_stack((step_inputs,chrp_inputs)), columns=['step','chrp'])
    # print("Computed inputs:")
    # print(df.head(30))


    # Write dataframe to csv file
    df.to_csv(dir_path + "/../input_signals/signals.csv", index=False)



      





if __name__ == "__main__":
    main()

