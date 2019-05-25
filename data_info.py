from load_dataset import load_dataset
from load_dataset import split_data
from load_dataset import accuracy_metric

import numpy as np
import pandas as pd

if __name__ == "__main__":
    Y_full = pd.read_csv('emittance_labels.csv')
    X_full = pd.read_csv('unit_cell_data.csv')

    total = pd.merge(X_full, Y_full, on="MPID")
    total_exps = len(total)
    print("Total Examples: " + str(total_exps))

    for i in range(10):
        col_name = "elem" + str(i)
        col_arr = np.array(total[col_name])
        num_elem_i = np.sum(col_arr > 0)
        print("Num elem " + str(i) + ": " + str(num_elem_i))
