import numpy as np
from sklearn.linear_model import LinearRegression

from load_dataset import load_dataset
from load_dataset import split_data

if __name__ == "__main__":

	X_train, Y_train, MPIDs_train, X_valid, Y_valid, MPIDs_valid, X_test, Y_test, MPIDs_test = split_data(load_dataset(threshold=-1))

	lin_clf = LinearRegression().fit(X_train, Y_train)
	predictions = lin_clf.predict(X_test)