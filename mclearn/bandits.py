""" Bandits algorithms. """

import numpy as np
from scipy.stats import beta
from .performance import compute_balanced_accuracy


class ActiveBandits(object):
	"""
	"""

    def __init__(self, classifier,
                 heuristics,
                 accuracy_fn=compute_balanced_accuracy,
                 initial_n=20,
                 training_size=100,
                 sample_size=20,
                 n_candidates=1,
                 verbose=False,
                 **kwargs):

        self.classifier = classifier
        self.heuristics = heuristics
        self.accuracy_fn = accuracy_fn
        self.initial_n = initial_n
        self.training_size = training_size
        self.current_training_size = 0
        self.n_candidates = n_candidates
        self.sample_size = sample_size
        self.verbose = verbose
        self.learning_curves_ = []
        n_heuristics = len(self.heuristics)
        self.n_wins = np.zeros(n_heuristics)
        self.n_trials = np.zeros(n_heuristics)


    def best_candidates(self, X_train, y_train, candidate_mask, classifier, n_candidates=1, **kwargs):
        """ Return the indices of the best candidates.

            Parameters
            ----------
            X_train : array
                The feature matrix of all the data points.

            y_train : array
                The target vector of all the data points.

            candidate_mask : boolean array
                The boolean array that tells us which data points the heuristic should look at.

            classifier : Classifier object
                A classifier object that will be used to make predictions.
                It should have the same interface as scikit-learn classifiers.

            n_candidates : int
                The number of best candidates to be selected at each iteration.

            **kwargs : other keyword arguments
                All other keyword arguments will be passed onto the heuristic function.


            Returns
            -------
            best_candidates : array
                The list of indices of the best candidates.
        """

        return self.heuristic(X_train, y_train, candidate_mask, classifier, n_candidates, **kwargs)


    def _random_sample(pool_size, train_mask, sample_size):
        """ Select a random sample from the pool.

            Parameters
            ----------
            pool_size : int
                The total number of data points (both queried and unlabelled)

            train_mask : boolean array
                The boolean array that tells us which data points are currently in the training set.

            sample_size : int
                The size of the random sample

            Returns
            -------
            candidate_mask : boolean array
                The boolean array that tells us which data points the heuristic should look at.
        """

        candidate_mask = -train_mask

        if 0 < self.sample_size < np.sum(candidate_mask):
            unlabelled_index = np.where(candidate_mask)[0]
            candidate_index = np.random.choice(unlabelled_index, self.sample_size, replace=False)
            candidate_mask = np.zeros(pool_size, dtype=bool)
            candidate_mask[candidate_index] = True

        return candidate_mask



    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """ Conduct active learning.

            Parameters
            ----------
            X_train : array
                The feature matrix of all the data points.

            y_train : array
                The target vector of all the data points.

            X_test : array
                If supplied, this will be used to compute an accuracy score for the learning curve.

            y_test : array
                If supplied, this will be used to compute an accuracy score for the learning curve.
        """

        pool_size = X_train.shape[0]
        n_features = X_train.shape[1]

        # boolean index of the samples which have been queried and are in the training set
        train_mask = np.zeros(pool_size, dtype=bool)

        # select an initial random sample from the pool and train the classifier
        sample = np.random.choice(np.arange(pool_size), self.initial_n, replace=False)
        train_mask[sample] = True
        self.classifier.fit(X_train[train_mask], y_train[train_mask])
        self.current_training_size += len(sample)

        # obtain the first data point of the learning curve
        if X_test is not None and y_test is not None:
            accuracy = self.accuracy_fn(self.classifier, X_test, y_test)
            self.learning_curves_.append(accuracy)

        # keep training the classifier until we have a desired sample size
        while np.sum(-unlabelled) < self.training_size:
            
            # select a random sample from the unlabelled pool
            candidate_mask = self._random_sample(pool_size, train_mask, self.sample_size)

            # pick the index of the best candidates
            best_candidates = self.best_candidates(X_train, y_train, candidate_mask,
                                                   classifier, n_candidates, train_mask, **kwargs)

            # retrain the classifier
            train_mask[best_candidates] = True
            self.classifier.fit(X_train[train_mask], y_train[train_mask])
            self.current_training_size += len(sample)

            # obtain the next data point of the learning curve
            if X_test is not None and y_test is not None:
                accuracy = self.accuracy_fn(self.classifier, X_test, y_test)
                self.learning_curves_.append(accuracy)

            # print progress after every 100 queries
            if verbose:
                if self.current_training_size % 1000 == 0:
                    print(self.current_training_size, end='')
                elif self.current_training_size % 100 == 0:
                    print('.', end='')



    def predict(self, X):
        """ Predict the target values of X given the model.

            Parameters
            ----------
            X : array
                The feature matrix

            Returns
            -------
            y : array
                Predicted values.
        """
        return classifier.predict(X)
