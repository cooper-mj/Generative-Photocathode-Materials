from load_dataset import load_dataset
from load_dataset import split_data
from load_dataset import accuracy_metric

import numpy as np
import pandas as pd
import pandasql as ps

if __name__ == "__main__":
    Y_full = pd.read_csv('emittance_labels.csv')
    X_full = pd.read_csv('unit_cell_data_16.csv')

    total = pd.merge(X_full, Y_full, on="MPID")
    total_exps = len(total)
    print("Total Examples: " + str(total_exps))

    for i in range(16):
        col_name = "elem" + str(i)
        col_arr = np.array(total[col_name])
        num_elem_i = np.sum(col_arr > 0)
        print("Num elem " + str(i) + ": " + str(num_elem_i))

    Y_arr = np.array(total["min emittance"])
    pos_exps = np.sum(Y_arr <= 0.2)
    print("Num pos exps (<= 0.2): " + str(pos_exps))

    inf_exps = np.sum(Y_arr == float('inf'))
    print("Num inf exps: " + str(inf_exps))

    total_arr = np.array(total)
    print(total_arr[0].shape)
    total_no_inf = np.array([total_arr[i] for i in range(total_exps) if total_arr[i, 48] != float('inf')])
    print(total_no_inf.shape)

    X_train, Y_train, MPIDs_train, X_valid, Y_valid, MPIDs_valid, X_test, Y_test, MPIDs_test = split_data(load_dataset(0.2))

    print("Training set information:")
    print("Positive examples: " + str(sum(Y_train)))
    print("Negative examples: " + str(len(Y_train) - sum(Y_train)))
