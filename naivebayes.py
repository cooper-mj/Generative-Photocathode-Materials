from sklearn.naive_bayes import BernoulliNB

from load_dataset import load_dataset
from load_dataset import split_data
from load_dataset import accuracy_metric

import numpy as np

if __name__ == "__main__":

	X_train, Y_train, MPIDs_train, X_valid, Y_valid, MPIDs_valid, X_test, Y_test, MPIDs_test = split_data(load_dataset("unit_cell_data_16.csv", 0.2))

	prior_0 = (len(Y_train) - sum(Y_train)) / float(len(Y_train))
	prior_1 = sum(Y_train) / float(len(Y_train))

	clf = BernoulliNB(class_prior=[prior_0, prior_1]).fit(X_train, Y_train)

	Y_valid_predictions = np.zeros(len(Y_valid))
	for i, example in enumerate(X_valid):
		Y_valid_predictions[i] = clf.predict(example.reshape(1, -1))

	accuracy_metric(Y_valid_predictions, Y_valid)
