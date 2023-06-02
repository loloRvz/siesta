#! /usr/bin/env python3


from siesta_mlp import *

from scipy import signal

### SCRIPT ###
def main():
    # Get all datasets
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/evaluation/5-mixd/*.csv')
    list_of_files = sorted(list_of_files)
    #list_of_files.reverse()
    

    # Load data from files
    times = []
    setpoints = []
    positions = []
    torques = []
    for path in list_of_files:
        print("Opening: ",path)

        # Prepare & plot dataset
        dataset = CSVDataset(path)
        dataset.preprocess()
        data = dataset.df.to_numpy()

        # Put values in arrays     
        times.append(data[:,TIME])
        setpoints.append(data[:,SETPOINT])
        positions.append(data[:,POSITION])
        torques.append(data[:,ACCELERATION_COMP] * LOAD_INERTIAS[dataset.load_id])


    # Compute difference to measurements
    min_len = min([pos_arr.size for pos_arr in positions])
    positions_equ = [pos_arr[:min_len] for pos_arr in positions]
    rmse = [mean_squared_error(positions_equ[0], p) for p in positions_equ]
    print(rmse)
    

    # Plot signals
    fig,ax=plt.subplots()
    ax.plot(times[0],setpoints[0],"--")
    for i in range(len(positions)):
        ax.plot(times[i],positions[i])
    ax.axhline(y=0, color='k')
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude [rad]")
    ax.legend(["Setpoint","Real system","PD Model","Trained NN Model"])                       
    plt.title("Position control comparison")
    plt.show()


if __name__ == "__main__":
    main()





# Compute lags
# for i in range(1,len(times)):
#     # Compute lag with cross correlation
#     corr = signal.correlate(setpoints[i], setpoints[0], mode='full')
#     lags = signal.correlation_lags(len(setpoints[i]), len(setpoints[0]), mode='full')
#     lag = lags[np.argmax(corr)]/400

#     # Get start lag
#     lag += times[i][0] - times[0][0]

#     #times[i] -= lag
#     print(lag, " seconds")

# times[1] -= 0
# times[2] -= 0