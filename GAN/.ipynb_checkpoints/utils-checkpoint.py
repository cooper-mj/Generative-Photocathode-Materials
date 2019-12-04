from numpy import genfromtxt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

''' load_dataset: Loads and merges cleaned-up data
	@param threshold: emittance threshold below which a data point is considered positive example
		Set threshold = -1 for non-binary models
	@return tuple of numpy arrays for MPIDs, features, and labels
'''
def load_dataset(filename):#, threshold=0.2):
	# Get data
	Y_full = pd.read_csv('../emittance_labels.csv')
	X_full = None
	if filename == "combined":
		X_unit_cell = pd.read_csv('../unit_cell_data_16.csv')
		X_avg = pd.read_csv('../material_average_data.csv')
		X_avg = pd.read_csv('../material_average_data_plus.csv')
		X_full = pd.merge(X_unit_cell, X_avg, on='MPID')
	else:
		X_full = pd.read_csv(filename)

	total = pd.merge(X_full, Y_full, on="MPID")

	# Random state below is a seed - change this when we go to run for real
	total = np.array(total.sample(frac=1, random_state=229).reset_index(drop=True))
	total = np.array([total[i] for i in range(len(total)) if total[i, -1] != float('inf')])

	MPIDs = np.array(total[:, 0])
	# X = np.array(total[:, 1:-1])

	# # Replace NaN/-1/0's with average col value as needed
	# nan_locs = np.isnan(X)
	# X[nan_locs] = -1
	# # print(len(X[0]))
	# # print(X)
	# _, colnum = X.shape

	# nonexistent = -1
	# if filename == 'material_average_data_plus.csv':
	# 	nonexistent = 0

	# for col in range(colnum):
	# 	adj_col = X[:, col]
	# 	mask = adj_col != nonexistent
	# 	mean = np.mean(adj_col * mask)
	# 	adj_col[adj_col == nonexistent] = mean

	filtered_total = np.array([total[i] for i in range(len(total)) if total[i, -1] < 0.5]) #0.5 is threshold value here
	X = filtered_total[:,1:-1]
	Y = np.array(filtered_total[:, -1])

	# Filter only the X's where the corresponding Y is less than 0.5



	# if filename == 'material_average_data.csv' or 'combined':
	# 	scaler = StandardScaler()
	# 	scaler.fit(X[-9:])
	# 	X[-9:] = scaler.transform(X[-9:])

	# Scale data
	# if filename == 'material_average_data.csv' or 'combined':
	# 	scaler = StandardScaler()
	# 	scaler.fit(X[-8:]) # scale everything except MPID and atomic number
	# 	X = scaler.transform(X)

	# if filename == 'material_average_data_plus.csv':
	# 	scaler = StandardScaler()
	# 	scaler.fit(X[-16:]) # scale everything except MPID
	# 	X = scaler.transform(X)

	# print(len(X))

	return (MPIDs, X, Y)

''' load_dataset: Loads and merges cleaned-up data
	@param tup: tuple of MPIs, X's, Y's as returned by load_dataset
	@return tuple of numpy arrays for training, validation, and test sets
'''
# def split_data(tup, train_split = 0.8, valid_split = 0.1, test_split = 0.1):

# 	MPIDs, X, Y = tup

# 	assert (train_split + valid_split + test_split == 1),"The proportion of data dedicated to train, validation, and test sets does not sum to 1."

# 	training_threshold = train_split
# 	valid_threshold = train_split + valid_split

# 	X_train = X[:int(len(X)*training_threshold)]
# 	Y_train = Y[:int(len(Y)*training_threshold)]
# 	MPIDs_train = MPIDs[:int(len(MPIDs)*training_threshold)]

# 	X_valid = X[int(len(X)*training_threshold):int(len(X)*valid_threshold)]
# 	Y_valid = Y[int(len(Y)*training_threshold):int(len(X)*valid_threshold)]
# 	MPIDs_valid = MPIDs[int(len(MPIDs)*training_threshold):int(len(X)*valid_threshold)]

# 	X_test = X[int(len(X)*valid_threshold):]
# 	Y_test = Y[int(len(Y)*valid_threshold):]
# 	MPIDs_test = MPIDs[int(len(MPIDs)*valid_threshold):]

# 	return (X_train, Y_train, MPIDs_train, X_valid, Y_valid, MPIDs_valid, X_test, Y_test, MPIDs_test)

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

	accuracy = (true_positives + true_negatives) / len(Y_actual)
	precision = true_positives / (true_positives + true_negatives)
	recall = true_positives / (true_positives + false_negatives)
	F1 = 2 * (precision * recall) / (precision + recall)

	print("Correctly Predicted Proportion : " + str(accuracy))
	print("Precision : " + str(precision))
	print("Recall : " + str(recall))
	print("F1 : " + str(F1))


def augment_data(X, Y, num_permutations):
	atoms = X[:, -64:]
	# print(X)
	# atoms = X[:, -6:]
	XT = atoms.T
	m, n = np.shape(XT)[0] // 2, np.shape(XT)[1]
	all_new_inputs = None
	all_labels = None

	for i in range(num_permutations):
		perm = XT.reshape(m, -1, n)[np.random.permutation(m)].reshape(-1,n)
		new_data = np.concatenate((X[:, :-64], perm.T), axis=1)
		# print(new_data)
		if i == 0:
			all_new_inputs = new_data
			all_labels = Y
		else:
			# print('Concatenating!')
			all_new_inputs = np.concatenate((all_new_inputs, new_data), axis=0)
			all_labels = np.concatenate((all_labels, Y), axis=0)
	return (np.concatenate((X, all_new_inputs), axis=0), np.concatenate((Y, all_labels), axis=0))
