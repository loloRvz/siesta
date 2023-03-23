#! /usr/bin/env python3


from siesta_mlp import *


### SCRIPT ###
def main():
    # Get all datasets
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/experiments/*.csv')
    list_of_files = sorted(list_of_files)
    list_of_files.reverse()

    time_s = 5000
    time_e = 5200

    fig,ax=plt.subplots()

    for set in range(0,4):
        path = list_of_files[set]
        print("Opening: ",path)

        # Prepare & plot dataset
        dataset = CSVDataset(path)
        dataset.preprocess()
        data = dataset.df.to_numpy()

        time_range = np.squeeze(np.where(np.logical_and(data[:,TIME]>=time_s, data[:,TIME]<=time_e)))

        delay = np.squeeze(np.where(data[time_range,SETPOINT] > -0))[0]

        times = data[:,TIME]
        currents = np.roll(data[:,CURRENT],-delay + 10)

        # print(data[time_range,CURRENT])
        ax.plot(times,currents)


    ax.axhline(y=0, color='k')
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Current[A]")
    ax.legend(["no load","224.5e-6 kg*m²","548.4e-6 kg*m²","772.9 e-6 kg*m²"])                        
    plt.title("Motor data reading @800Hz")
    plt.show()


if __name__ == "__main__":
    main()