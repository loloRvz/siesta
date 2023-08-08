#! /usr/bin/env python3


from siesta_mlp import *
from generate_input_signals import *


### SCRIPT ###
def main():
    dir_path = os.path.dirname(os.path.realpath(__file__)) # Directory of this file

    ### SIMULATION SETUP ###

    # Params
    T1 = 0          # [s]
    T2 = 55         # [s]
    F = 400         # [Hz]
    N = (T2-T1)*F   # []
    START = 0    # [kRPM]

    # Input signal
    input_type = FLIT
    input_dataset = CSVDataset("../data/input_signals/signals.csv")
    input_np = input_dataset.df.to_numpy(dtype=np.float64)
    setpoint = input_np[:N,input_type]
    time = np.linspace(T1,T2, N)

    # Data arrays
    signal_names = []
    times = []
    setpoints = []
    positions = []


    ### MEASURED DATA ###

    # Get measured signal (true)
    eval_data_dirs = ["0-step","1-ramp","2-chrp","3-flit","4-nois","5-mixd","6-flitreal"]
    eval_data_path = sorted(glob.glob("../data/evaluation/" + eval_data_dirs[input_type] + "/*.csv"))[0]
    print("Opening measurement data: ", eval_data_path)
    eval_dataset = CSVDataset(eval_data_path)
    eval_dataset.preprocess()
    eval_np = eval_dataset.df.to_numpy(dtype=np.float64)
    setpoint = np.interp(time, eval_np[:N,TIME], eval_np[:N,SETPOINT])

    # Save data to arrays
    signal_names.append("Measurement")
    times.append(eval_np[:,TIME])
    setpoints.append(eval_np[:,SETPOINT])
    positions.append(eval_np[:,POSITION])


    ### P-CONTROLLER ###

    # Params
    K_P = 2
    K_D = 0.05
    torque_max = 2e10
    torque_min = -torque_max
    delta_t = 1/FREQUENCY

    # Init state
    position = np.zeros(time.shape)
    velocity = np.zeros(time.shape)

    # Simulate response
    for i in range(N-1):

        torque = K_P*(setpoint[i] - position[i]) - K_D * velocity[i]
        if torque > torque_max: cmd = torque_max
        if torque < torque_min: cmd = torque_min

        cmd = torque / LOAD_INERTIA

        velocity[i+1] = velocity[i] + cmd * delta_t
        position[i+1] = position[i] + velocity[i]*delta_t + 0.5*cmd*delta_t**2

    # Save data to arrays
    signal_names.append("PD Model")
    times.append(time)
    setpoints.append(setpoint)
    positions.append(position)


    ### NEURAL NETWORK MODEL ###

    # Load pre-trained model
    model_dirs = dir_path + "/../data/models/" + "23-05-30--14-53-36_flit-PHL05big/"
    list_of_models = glob.glob(model_dirs + '*.pt')
    list_of_models = sorted(list_of_models)

    # Model parameters
    h_len = 5

    delta_t = 1/FREQUENCY

    # Simulate all models
    for model_dir in list_of_models[:]:
        print("Opening model:", model_dir[-43:])
        model = torch.jit.load(model_dir)

        # Init state
        position = np.zeros(time.shape)

        prev_set = np.ones(1)*START
        prev_pos = np.ones(h_len)*START

        # Simulate response
        for i in range(N-1):
            in_tens = torch.tensor(np.concatenate((prev_set,prev_pos), axis=None), dtype=torch.float64)
            out_tens = model(in_tens)
            vel = out_tens.item()

            prev_set = np.roll(prev_set,1)
            prev_pos = np.roll(prev_pos,1)
            
            prev_set[0] = setpoint[i]
            prev_pos[0] = position[i] + vel*delta_t

            position[i+1] = prev_pos[0]

        # Plot validation dataset to time
        signal_names.append("Epoch: " + model_dir[-7:-3])
        times.append(time)
        setpoints.append(setpoint)
        positions.append(position)


    ### DATA PROCESSING ###

    num_signals = len(positions)

    # Sync measurements times
    min_time = 2
    max_time = min([times_arr[-1] for times_arr in times]) - 2
    positions = [positions[i][np.logical_and(min_time < times[i],times[i] < max_time)]  for i in range(num_signals)]
    setpoints = [setpoints[i][np.logical_and(min_time < times[i],times[i] < max_time)]  for i in range(num_signals)]
    times = [times[i][np.logical_and(min_time < times[i],times[i] < max_time)]  for i in range(num_signals)]

    # Interpolate signals to sync times
    positions_interp = [np.interp(times[0], times[i], positions[i]) for i in range(num_signals)]
    setpoints_interp = [np.interp(times[0], times[i], setpoints[i]) for i in range(num_signals)]
    errors_interp = [(np.square(positions_interp[0]-positions_interp[i])) for i in range(num_signals)]

    # Compute errors to measurements
    rmse = [math.sqrt(mean_squared_error(positions_interp[0], p)) for p in positions_interp]
    rmse_set = [math.sqrt(mean_squared_error(setpoints_interp[0], s)) for s in setpoints_interp]


    ### PLOTTING ###

    # Print RMSE of all signals
    print("RMSE")
    for i in range(num_signals):
        print("Signal: ",signal_names[i],", RMSE: ",rmse[i])

    # Plot signals
    plt.figure(1,figsize=(7,5))
    plt.plot(times[0],setpoints[0],linestyle="dashed",linewidth=2)            # Plot setpoint
    plt.plot(times[0],positions_interp[0],linewidth=2)    # Plot measured values
    # Plot NN models
    for i in range(1,num_signals):
        plt.plot(times[0],positions_interp[i],linewidth=1.5)
    plt.axhline(y=0, color='k')
    # for i in range(1,num_signals):
    #     plt.plot(times[0],errors_interp[i],"--",color="C"+str(i+1))
    leg = ["Setpoint","Measurement","PD Model","NN Model"]
    plt.xlabel("Time [s]")
    plt.ylabel("Position [rad]")
    plt.ylim([-0.5, +0.5])
    plt.legend(leg)                         
    plt.title("Position Control")
    plt.xlim([9.0,10.0])
    plt.ylim([-0.07,0.07])

    plt.figure(2,figsize=(7,5))
    plt.plot([int(epoch[-4:]) for epoch in signal_names[2:]],rmse[2:])
    plt.hlines(rmse[1],0,int(signal_names[-1][-4:]),color="green")
    plt.legend(leg[2:])
    plt.xlabel("Model Epoch")
    plt.ylabel("RMSE")
    plt.ylim([0, 0.005])
    plt.title("Model Performance")

    try:
        plt.figure(3,figsize=(2.5,5))
        bars = (["PD Model","NN Model"])
        x_pos = [0,1]
        plt.bar(x_pos, rmse[1:], width = 0.5, align='center')
        plt.subplots_adjust(left=0.35)
        plt.xticks(x_pos, bars)
        plt.xlabel("Model Type")
        plt.ylabel("RMSE")
        plt.title("Model Performance")
    except:
        print("Error plotting bar plot")

    
    plt.show()





if __name__ == "__main__":
    main()