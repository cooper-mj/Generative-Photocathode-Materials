from sklearn.svm import SVC

from load_dataset import load_dataset
from load_dataset import split_data
from load_dataset import accuracy_metric

import numpy as np

if __name__ == "__main__":
	X_train, Y_train, MPIDs_train, X_valid, Y_valid, MPIDs_valid, X_test, Y_test, MPIDs_test = split_data(load_dataset(0.2))

	print("Training set information:")
	print("Positive examples: " + str(sum(Y_train)))
	print("Negative examples: " + str(len(Y_train) - sum(Y_train)))

	clf = SVC(gamma='auto').fit(X_train, Y_train)

	Y_valid_predictions = np.zeros(len(Y_valid))
	for i, example in enumerate(X_valid):
		Y_valid_predictions[i] = clf.predict(example.reshape(1, -1))

	accuracy_metric(Y_valid_predictions, Y_valid)
