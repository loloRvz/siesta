#! /usr/bin/env python3


from siesta_mlp import *


### SCRIPT ###
def main():
    # Open measured data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../data/*.csv')
    cwd = os.getcwd()
    path = max(list_of_files, key=os.path.getctime)
    #path = '../data/2023-02-22--15-00-04_dataset.csv'
    print("Opening: ",path)

    # Prepare & plot dataset
    dataset = CSVDataset(path)
    dataset.preprocess()
    dataset.plot_data()


if __name__ == "__main__":
    main()