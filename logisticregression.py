
from sklearn.linear_model import LogisticRegression
from numpy import genfromtxt
import numpy as np


def load_dataset():
	# Add code here to pull the dataset
	
	# Get labels
	Y_full = genfromtxt('emittance_labels.csv', delimiter=',')

	# Remove nan objects at the top
	Y_full = Y_full[1:]

	# Separate by column
	Y_labels = Y_full[:,0]
	Y = Y_full[:,1]

	# Round to the nearest tenth
	for i in range(len(Y)):
		Y[i] = round(Y[i], 2)

	return

if __name__ == "__main__":
	load_dataset()

