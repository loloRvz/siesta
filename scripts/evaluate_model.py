#! /usr/bin/env python3


from siesta_mlp import *


### SCRIPT ###
def main():
    # Directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))


    # Open measured data
    list_of_files = glob.glob(dir_path + '/../data/experiments/evaluation/*.csv')
    list_of_files = sorted(list_of_files)
    list_of_files.reverse()


    # Load trained model
    model_dir = dir_path + "/../data/models/"+"23-03-29--10-10-00_400Hz-L9-nois-PHL5_Ta"  + "/delta_300.pt"
    model = torch.jit.load(model_dir)

    h_len = 5

    for path in list_of_files:
        print("Opening: ",path)    
        dataset = CSVDataset(path)
        dataset.preprocess()
        dataset.prepare_data(h_len, torque_est=1)
        full_dl = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
        mse = evaluate_model(full_dl, model, NILL = True)
        print('MSE: %.5f, RMSE: %.5f' % (mse, np.sqrt(mse)))
        #plot_model_predictions(dataset, model, np.sqrt(mse), NILL = False)



if __name__ == "__main__":
    main()