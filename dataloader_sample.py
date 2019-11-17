import numpy as np
from sklearn.linear_model import LinearRegression

from load_dataset import load_dataset

if __name__ == "__main__":

	_, X, Y = load_dataset("unit_cell_data_16.csv")
	print(X)