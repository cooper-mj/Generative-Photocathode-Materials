from numpy import genfromtxt
import numpy as np
import pandas as pd

''' load_dataset: Loads and merges cleaned-up data
	@param threshold: emittance threshold below which a data point is considered positive example
		Set threshold = -1 for non-binary models
	@return tuple of numpy arrays for MPIDs, features, and labels
'''
def load_dataset(threshold=0.2):
	# Get data	
	Y_full = pd.read_csv('emittance_labels.csv')
	X_full = pd.read_csv('unit_cell_data.csv')

	total = pd.merge(X_full, Y_full, on="MPID")

	# Random state below is a seed - change this when we go to run for real
	total = total.sample(frac=1, random_state=229).reset_index(drop=True)

	MPIDs = np.array(total[['MPID']])

	X = np.array(total.iloc[:, 1:48])

	# print(X)
	Y = np.array(total[["min emittance"]])
	# print(Y)

	if threshold != -1:
		Y = [1 if y_i < threshold else 0 for y_i in Y]

	return (MPIDs, X, Y)

''' load_dataset: Loads and merges cleaned-up data
	@param tup: tuple of MPIs, X's, Y's as returned by load_dataset
	@return tuple of numpy arrays for training, validation, and test sets
'''
def split_data(tup, train_split = 0.6, valid_split = 0.2, test_split = 0.2):

	MPIDs, X, Y = tup

	assert (train_split + valid_split + test_split == 1),"The proportion of data dedicated to train, validation, and test sets does not sum to 1."

	training_threshold = train_split
	valid_threshold = train_split + valid_split

	X_train = X[:int(len(X)*training_threshold)]
	Y_train = Y[:int(len(Y)*training_threshold)]
	MPIDs_train = MPIDs[:int(len(MPIDs)*training_threshold)]

	X_valid = X[int(len(X)*training_threshold):int(len(X)*valid_threshold)]
	Y_valid = Y[int(len(Y)*training_threshold):int(len(X)*valid_threshold)]
	MPIDs_valid = MPIDs[int(len(MPIDs)*training_threshold):int(len(X)*valid_threshold)]

	X_test = X[int(len(X)*valid_threshold):]
	Y_test = Y[int(len(Y)*valid_threshold):]
	MPIDs_test = MPIDs[int(len(MPIDs)*valid_threshold):]

	return (X_train, Y_train, MPIDs_train, X_valid, Y_valid, MPIDs_valid, X_test, Y_test, MPIDs_test)

def accuracy_metric(Y_predictions, Y_actual):
	true_positives = 0.0
	true_negatives = 0.0
	false_positives = 0.0
	false_negatives = 0.0

	for i, prediction in enumerate(Y_predictions):
		if Y_actual[i] == 1 and Y_predictions[i] == 1:
			true_positives += 1
		if Y_actual[i] == 0 and Y_predictions[i] == 0:
			true_negatives += 1
		if Y_actual[i] == 1 and Y_predictions[i] == 0:
			false_negatives += 1
		if Y_actual[i] == 0 and Y_predictions[i] == 1:
			false_positives += 1

	print("Correctly Predicted Proportion : " + str((true_positives + true_negatives) / len(Y_actual)))
	print("Precision : " + str(true_positives / (true_positives + true_negatives)))
	print("Recall : " + str(true_positives / (true_positives + false_negatives)))


