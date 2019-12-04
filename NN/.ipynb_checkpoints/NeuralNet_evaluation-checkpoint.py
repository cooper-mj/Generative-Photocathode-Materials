from load_dataset import load_dataset
from load_dataset import split_data
import pickle
from sklearn.neural_network import MLPClassifier
import numpy as np
from load_dataset import accuracy_metric

if __name__ == "__main__":
	filename = 'unit_cell_data_16.csv'

	X_train, Y_train, MPIDs_train, X_valid, Y_valid, MPIDs_valid, X_test, Y_test, MPIDs_test = split_data(load_dataset(filename, 0.2))

	clf = pickle.load(open('NN_evaluator.sav', 'rb'))

	Y_valid_predictions = np.zeros(len(Y_valid))
	for i, example in enumerate(X_valid):
		Y_valid_predictions[i] = clf.predict(example.reshape(1, -1))
	accuracy_metric(Y_valid_predictions, Y_valid)
