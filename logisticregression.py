
from sklearn.linear_model import LogisticRegression

from load_dataset import load_dataset
from load_dataset import split_data

if __name__ == "__main__":

	X_train, Y_train, MPIDs_train, X_valid, Y_valid, MPIDs_valid, X_test, Y_test, MPIDs_test = split_data(load_dataset())

	clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter = 10000000).fit(X_train, Y_train)

	for example in X_valid:
		clf.predict()

