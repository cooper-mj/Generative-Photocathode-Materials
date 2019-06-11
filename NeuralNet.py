from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from load_dataset import load_dataset
from load_dataset import split_data
from load_dataset import accuracy_metric
from load_dataset import augment_data

import numpy as np
import itertools as it

# Try excluding 1 element at a time
def run_experiment(clf, X_train, Y_train, X_valid, Y_valid):
	num_examples, num_features = np.shape(X_train)
	num_tests = np.shape(X_valid)[0]
	# to_exclude = [2, 3, 5, 7, 12, 13, 15]
	to_exclude = [0, 1, 4, 6, 8, 9, 10, 11, 14]
	X_train_trunc = np.zeros((num_examples, num_features - 1))
	X_valid_trunc = np.zeros((num_tests, num_features - 1))
	# print(X_train)
	for i in to_exclude:
		print('Now excluding feature %d' % i)
		X_train_trunc[:, :i] = X_train[:, :i]
		X_train_trunc[:, i:] = X_train[:, i + 1:]

		X_valid_trunc[:, :i] = X_valid[:, :i]
		X_valid_trunc[:, i:] = X_valid[:, i + 1:]

		clf.fit(X_train_trunc, Y_train)
		Y_valid_predictions = np.zeros(len(Y_valid))
		for i, example in enumerate(X_valid_trunc):
			Y_valid_predictions[i] = clf.predict(example.reshape(1, -1))

		accuracy_metric(Y_valid_predictions, Y_valid)

if __name__ == "__main__":
	# filename = 'material_average_data_plus.csv'
	# filename = 'combined'
	filename = 'unit_cell_data_16.csv'

	X_train, Y_train, MPIDs_train, X_valid, Y_valid, MPIDs_valid, X_test, Y_test, MPIDs_test = split_data(load_dataset(filename, 0.2))

	# If data augmentation needed
	X_train, Y_train = augment_data(X_train, Y_train, 10)

	print("Training set information:")
	pos_ex = sum(Y_train)
	print("Positive examples: " + str(pos_ex))
	print("Negative examples: " + str(len(Y_train) - pos_ex))

	# print(X_train)

	# scaler = StandardScaler()
	# scaler.fit(X_train)
	# X_train = scaler.transform(X_train)
	# X_valid = scaler.transform(X_valid)
	# X_test = scaler.transform(X_test)

	clf = MLPClassifier(hidden_layer_sizes=(64, 128, 256, 512, 256, 128, 64), learning_rate_init=0.0008, max_iter=400, n_iter_no_change=10, verbose=True).fit(X_train, Y_train)

	Y_valid_predictions = np.zeros(len(Y_valid))
	for i, example in enumerate(X_valid):
		Y_valid_predictions[i] = clf.predict(example.reshape(1, -1))

    clf = MLPClassifier(hidden_layer_sizes=(128, 256, 512, 512, 256, 128, 64), learning_rate_init = 0.0008, max_iter=300, n_iter_no_change=15, verbose=True).fit(X_train, Y_train)
	accuracy_metric(Y_valid_predictions, Y_valid)

	# clf = MLPClassifier(hidden_layer_sizes=(64, 128, 256, 128, 64), learning_rate_init=0.0008, max_iter=400, n_iter_no_change=10, verbose=False)
	# run_experiment(clf, X_train, Y_train, X_valid, Y_valid)

	# X = np.asarray([[0,0,1,2,3,4,5,6],[0,0,2,3,4,5,6,7],[0,0,3,4,5,6,7,8],[0,0,4,5,6,7,8,9]])
	# Y = [1,1,0,1]
