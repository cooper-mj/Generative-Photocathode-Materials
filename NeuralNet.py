from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from load_dataset import load_dataset
from load_dataset import split_data
from load_dataset import accuracy_metric
from load_dataset import augment_data

import numpy as np
import itertools as it
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

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
	# X_train, Y_train = augment_data(X_train, Y_train, 10)

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

	clf = MLPClassifier(hidden_layer_sizes=(64, 128, 256, 512, 256, 128, 64), alpha=0.5, tol=0.001, learning_rate_init=0.0008, max_iter=400, n_iter_no_change=10, verbose=True).fit(X_train, Y_train)
	# clf = MLPClassifier(hidden_layer_sizes=(64, 128, 64), learning_rate_init=0.0008, max_iter=400, n_iter_no_change=10, verbose=True).fit(X_train, Y_train)
	Y_valid_predictions = np.zeros(len(Y_valid))
	for i, example in enumerate(X_valid):
		Y_valid_predictions[i] = clf.predict(example.reshape(1, -1))

	# clf = MLPClassifier(hidden_layer_sizes=(128, 256, 512, 256, 128, 64), learning_rate_init = 0.0008, max_iter=300, n_iter_no_change=15, verbose=True) #.fit(X_train, Y_train)
	accuracy_metric(Y_valid_predictions, Y_valid)
	# title = "Learning Curves (Neural Network)"

	# estimator = GaussianNB()
	# plot_learning_curve(clf, title, X_train, Y_train, ylim=(0.0, 1.01), n_jobs=4)
	# plt.show()
