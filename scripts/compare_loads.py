#! /usr/bin/env python3


from siesta_mlp import *


### SCRIPT ###
def main():
    # Get all datasets
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/experiments/training/*.csv')
    list_of_files = sorted(list_of_files)
    list_of_files.reverse()

    time_s = 2.4
    time_e = 2.6

    fig,ax=plt.subplots()

    for set in range(0,2):
        path = list_of_files[set]
        print("Opening: ",path)

        # Prepare & plot dataset
        dataset = CSVDataset(path)
        dataset.preprocess()
        data = dataset.df.to_numpy()

        time_range = np.squeeze(np.where(np.logical_and(data[:,TIME]>=time_s, data[:,TIME]<=time_e)))

        delay = np.squeeze(np.where(data[time_range,SETPOINT] > 0))[0]
        #delay = 0
        print("File Nr.: ", set, " delay: ", delay)

        times = data[:,TIME]
        values = np.roll(data[:,ACCELERATION_COMP],-delay)

        ax.plot(times,values)


    ax.axhline(y=0, color='k')
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude[]")
    ax.legend(["no load","224.5e-6 kg*m²","548.4e-6 kg*m²","772.9 e-6 kg*m²","drone arm"]) 
    ax.legend(["400Hz","800Hz"])                        
    plt.title("Motor data reading @800Hz")
    plt.show()


if __name__ == "__main__":
    main()