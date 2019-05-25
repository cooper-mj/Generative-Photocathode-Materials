THRESHOLD = 0.4

from numpy import genfromtxt
import numpy as np
import pandas as pd

def load_dataset():
	# Add code here to pull the dataset
	
	# Get labels
	
	Y_full = pd.read_csv('emittance_labels.csv')

	X_full = pd.read_csv('unit_cell_data.csv')

	total = pd.merge(X_full, Y_full, on="MPID")

	# Random state below is a seed - change this when we go to run for real
	total = total.sample(frac=1, random_state=229).reset_index(drop=True)

	MPIDs = np.array(total[['MPID']])

	X = np.array(total.iloc[:, 1:48])

	Y = np.array(total[["min emittance"]])

	Y = [1 if y_i < THRESHOLD else 0 for y_i in Y]
	return (MPIDs, X, Y)


def split_data(tup):

	MPIDs, X, Y = tup

	X_train = X[:int(len(X)*0.6)]
	Y_train = Y[:int(len(Y)*0.6)]
	MPIDs_train = MPIDs[:int(len(MPIDs)*0.6)]

	X_valid = X[int(len(X)*0.6):int(len(X)*0.8)]
	Y_valid = Y[int(len(Y)*0.6):int(len(X)*0.8)]
	MPIDs_valid = MPIDs[int(len(MPIDs)*0.6):int(len(X)*0.8)]

	X_test = X[int(len(X)*0.8):]
	Y_test = Y[int(len(Y)*0.8):]
	MPIDs_test = MPIDs[int(len(MPIDs)*0.8):]

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


