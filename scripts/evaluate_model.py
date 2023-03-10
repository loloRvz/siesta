#! /usr/bin/env python3


from siesta_mlp import *


### SCRIPT ###
def main():
    # Directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Load trained model
    model_dir = dir_path + "/../data/models/"+"23-03-08--09-21-01_L1-mixd-PHL5" + "/delta_300.pt"
    model = torch.jit.load(model_dir)

    # Open measured data
    list_of_files = glob.glob(dir_path + '/../data/experiments/*.csv')
    list_of_files = sorted(list_of_files)
    list_of_files.reverse()

    for path in list_of_files:
        print("Opening: ",path)    
        dataset = CSVDataset(path)
        dataset.preprocess()
        dataset.prepare_data(HIST_LENGTH)
        full_dl = DataLoader(dataset, batch_size=1024, shuffle=False, pin_memory=True)
        mse = evaluate_model(full_dl, model)
        print('MSE: %.3f, RMSE: %.3f' % (mse, np.sqrt(mse)))
        plot_model_predictions(dataset, model)



if __name__ == "__main__":
    main()