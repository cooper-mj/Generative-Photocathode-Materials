from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from load_dataset import load_dataset
from load_dataset import split_data
from load_dataset import accuracy_metric

import numpy as np

if __name__ == "__main__":
	filename = 'material_average_data_plus.csv'

	X_train, Y_train, MPIDs_train, X_valid, Y_valid, MPIDs_valid, X_test, Y_test, MPIDs_test = split_data(load_dataset(filename, 0.2))

	print("Training set information:")
	print("Positive examples: " + str(sum(Y_train)))
	print("Negative examples: " + str(len(Y_train) - sum(Y_train)))

	# scaler = StandardScaler()
	# scaler.fit(X_train)
	# X_train = scaler.transform(X_train)
	# X_valid = scaler.transform(X_valid)
	# X_test = scaler.transform(X_test)

	clf = MLPClassifier(hidden_layer_sizes=(64, 128, 256, 128, 64), learning_rate_init=0.0008, max_iter=400, n_iter_no_change=10, verbose=True).fit(X_train, Y_train)

	Y_valid_predictions = np.zeros(len(Y_valid))
	for i, example in enumerate(X_valid):
		Y_valid_predictions[i] = clf.predict(example.reshape(1, -1))

	accuracy_metric(Y_valid_predictions, Y_valid)
