""" The main routine of all active learning algorithms. """

import pickle
import numpy as np
from numpy.random import RandomState
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle

from .heuristics import random_h
from .performance import compute_balanced_accuracy


class BaseActive:
    """ Base class for active learning. """

    def __init__(self, classifier, best_heuristic=None, accuracy_fn=compute_balanced_accuracy,
                 initial_n=20, training_size=100, sample_size=20, n_candidates=1,
                 verbose=False, random_state=None, pool_random_state=None, **h_kwargs):

        self.classifier = classifier
        self.best_heuristic = best_heuristic
        self.accuracy_fn = accuracy_fn
        self.initial_n = initial_n
        self.training_size = training_size
        self.current_training_size = 0
        self.n_candidates = n_candidates
        self.sample_size = sample_size
        self.verbose = verbose
        self.learning_curve_ = []
        self.h_kwargs = h_kwargs
        self.candidate_selections = []
        self.rng = RandomState(random_state)
        self.pool_rng = RandomState(pool_random_state)


    def _random_sample(self, pool_size, train_mask, sample_size):
        """ Select a random sample from the pool.

            Parameters
            ----------
            pool_size : int
                The total number of data points (both queried and unlabelled).

            train_mask : boolean array
                The boolean array that tells us which points are currently in the training set.

            sample_size : int
                The size of the random sample.

            Returns
            -------
            candidate_mask : boolean array
                The boolean array that tells us which data points the heuristic should examine.
        """

        candidate_mask = -train_mask

        if 0 < self.sample_size < np.sum(candidate_mask):
            unlabelled_index = np.where(candidate_mask)[0]
            candidate_index = self.rng.choice(unlabelled_index, self.sample_size, replace=False)
            candidate_mask = np.zeros(pool_size, dtype=bool)
            candidate_mask[candidate_index] = True

        return candidate_mask


    def _select_heuristic(self):
        """ Choose a heuristic to be used (useful in bandits active learning). """

        return None


    def _store_results(self, accuracy):
        """ Store results at the end of an iteration. """

        self.learning_curve_.append(accuracy)


    def _print_progress(self):
        """ Print out current progress. """
        if self.current_training_size % 1000 == 0:
            print(self.current_training_size, end='')
        elif self.current_training_size % 100 == 0:
            print('.', end='')


    def select_candidates(self, X, y, candidate_mask, train_mask):
        """ Return the indices of the best candidates.

            Parameters
            ----------
            X : array
                The feature matrix of all the data points.

            y : array
                The target vector of all the data points.

            candidate_mask : boolean array
                The boolean array that tells us which data points the heuristic should examine.

            n_candidates : int
                The number of best candidates to be selected at each iteration.

            **h_kwargs : other keyword arguments
                All other keyword arguments will be passed onto the heuristic function.

            Returns
            -------
            best_candidates : array
                The list of indices of the best candidates.
        """

        return self.best_heuristic(X=X, y=y, candidate_mask=candidate_mask,
                                   train_mask=train_mask, classifier=self.classifier,
                                   n_candidates=self.n_candidates, random_state=self.pool_rng.randint(1000),
                                   **self.h_kwargs)


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
        sample = self.rng.choice(np.arange(pool_size), self.initial_n, replace=False)
        self.candidate_selections += list(sample)
        train_mask[sample] = True
        self.classifier.fit(X_train[train_mask], y_train[train_mask])
        self.current_training_size += len(sample)

        # obtain the first data point of the learning curve
        if X_test is not None and y_test is not None:
            accuracy = self.accuracy_fn(self.classifier, X_test, y_test)
            self.learning_curve_.append(accuracy)

        # keep training the classifier until we have a desired sample size
        while np.sum(train_mask) < self.training_size:
            
            # select a random sample from the unlabelled pool
            candidate_mask = self._random_sample(pool_size, train_mask, self.sample_size)

            # select the heuristic to be used
            self._select_heuristic()

            # pick the index of the best candidates
            best_candidates = self.select_candidates(X_train, y_train, candidate_mask, train_mask)
            self.candidate_selections += list(best_candidates)

            # retrain the classifier
            train_mask[best_candidates] = True
            self.classifier.fit(X_train[train_mask], y_train[train_mask])
            self.current_training_size += len(best_candidates)

            # obtain the next data point of the learning curve
            if X_test is not None and y_test is not None:
                accuracy = self.accuracy_fn(self.classifier, X_test, y_test)
                self._store_results(accuracy)

            # print progress after every 100 queries
            if self.verbose:
                self._print_progress()

            assert self.current_training_size == np.sum(train_mask), \
                   'Mismatch detected in the training size. Check your heuristic.'


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
        return self.classifier.predict(X)



class ActiveLearner(BaseActive):
    """ Active Learner

        Parameters
        ----------
        classifier : Classifier object
            A classifier object that will be used to train and test the data.
            It should have the same interface as scikit-learn classifiers.

        heuristic : function
            This is the function that implements the active learning rule. Given a set
            of training candidates and the classifier as inputs, the function will
            return index array of candidate(s) with the highest score(s).

        accuracy_fn : function
            Given a trained classifier, a test set, and a test oracle, this function
            will return the accuracy rate.

        initial_n : int
            The number of samples that the active learner will randomly select at the beginning
            to get the algorithm started.

        training_size : int
            The total number of samples that the active learner will query.

        n_candidates : int
            The number of best candidates to be selected at each iteration.

        sample_size : int
            At each iteration, the active learner will pick a random of sample of examples.
            It will then compute a score for each of example and query the one with the
            highest score according to the active learning rule. If sample_size is set to 0,
            the entire training pool will be sampled (which can be inefficient with large
            datasets).

        verbose : boolean
            If set to True, progress is printed to standard output after every 100 iterations.

        **kwargs : other keyword arguments
            All other keyword arguments will be passed onto the heuristic function.
        

        Attributes
        ----------
        learning_curves_ : array
            Every time the active learner queries the oracle, it will re-train the classifier
            and run it on the test data to get an accuracy rate. The learning curve is
            simply the array containing all of these accuracy rates.
    """

    def __init__(self, classifier, heuristic=random_h, accuracy_fn=compute_balanced_accuracy,
                 initial_n=20, training_size=100, sample_size=20, n_candidates=1,
                 verbose=False, **h_kwargs):

        super().__init__(classifier=classifier, best_heuristic=heuristic,
                         accuracy_fn=accuracy_fn, initial_n=initial_n,
                         training_size=training_size, sample_size=sample_size,
                         n_candidates=n_candidates, verbose=verbose, **h_kwargs)


class ActiveBandit(BaseActive):
    """ Active Learner Bandit

        Parameters
        ----------
        classifier : Classifier object
            A classifier object that will be used to train and test the data.
            It should have the same interface as scikit-learn classifiers.

        heuristic : function
            This is the function that implements the active learning rule. Given a set
            of training candidates and the classifier as inputs, the function will
            return index array of candidate(s) with the highest score(s).

        accuracy_fn : function
            Given a trained classifier, a test set, and a test oracle, this function
            will return the accuracy rate.

        initial_n : int
            The number of samples that the active learner will randomly select at the beginning
            to get the algorithm started.

        training_size : int
            The total number of samples that the active learner will query.

        n_candidates : int
            The number of best candidates to be selected at each iteration.

        sample_size : int
            At each iteration, the active learner will pick a random of sample of examples.
            It will then compute a score for each of example and query the one with the
            highest score according to the active learning rule. If sample_size is set to 0,
            the entire training pool will be sampled (which can be inefficient with large
            datasets).

        verbose : boolean
            If set to True, progress is printed to standard output after every 100 iterations.

        **kwargs : other keyword arguments
            All other keyword arguments will be passed onto the heuristic function.
        

        Attributes
        ----------
        learning_curves_ : array
            Every time the active learner queries the oracle, it will re-train the classifier
            and run it on the test data to get an accuracy rate. The learning curve is
            simply the array containing all of these accuracy rates.
    """

    def __init__(self, classifier, heuristics, accuracy_fn=compute_balanced_accuracy,
                 initial_n=20, training_size=100, sample_size=20, n_candidates=1,
                 verbose=False, prior_mu=0, prior_sigma=0.02,
                 likelihood_sigma=0.02, **h_kwargs):

        super().__init__(classifier=classifier,
                         accuracy_fn=accuracy_fn, initial_n=initial_n,
                         training_size=training_size, sample_size=sample_size,
                         n_candidates=n_candidates, verbose=verbose, **h_kwargs)

        self.heuristics = heuristics
        self.n_heuristics = len(heuristics)
        self.best_heuristic_idx = None
        self.heuristic_selection = []

        self.prior_mus = np.full(self.n_heuristics, prior_mu)
        self.prior_sigmas = np.full(self.n_heuristics, prior_sigma)
        self.likelihood_sigmas = np.full(self.n_heuristics, likelihood_sigma)

        self.all_prior_mus = [self.prior_mus.copy()]
        self.all_prior_sigmas = [self.prior_sigmas.copy()]


    def _select_heuristic(self):
        """ Choose a heuristic to be used (useful in bandits active learning). """

        # take a sample of rewards from the current prior of heuristics
        sample_rewards = np.random.normal(self.prior_mus, self.prior_sigmas)

        # select the heuristic that has the highest reward sample value
        self.best_heuristic_idx = np.argmax(sample_rewards)
        self.best_heuristic = self.heuristics[self.best_heuristic_idx]
        self.heuristic_selection.append(self.best_heuristic_idx)


    def _store_results(self, accuracy):
        """ Store results at the end of an iteration. """

        self.learning_curve_.append(accuracy)

        # update reward prior with the change in accuracy rate
        delta = self.learning_curve_[-1] - self.learning_curve_[-2]
        mu_0 = self.prior_mus[self.best_heuristic_idx]
        sigma_0 = self.prior_sigmas[self.best_heuristic_idx]
        sigma = self.likelihood_sigmas[self.best_heuristic_idx]

        self.prior_mus[self.best_heuristic_idx] = (mu_0 * sigma + delta * sigma_0) / (sigma + sigma_0)
        self.prior_sigmas[self.best_heuristic_idx] = (sigma_0 * sigma) / (sigma + sigma_0)

        self.all_prior_mus.append(self.prior_mus.copy())
        self.all_prior_sigmas.append(self.prior_sigmas.copy())


def run_active_learning_expt(X, y, kfold, classifier, committee, heuristics, pickle_paths,
    initial_n=50, training_size=300, sample_size=300, verbose=True,
    committee_samples=300, pool_n=300, C=1):
    """ Run an active learning experiment
    """

    for heuristic, pickle_path in zip(heuristics, pickle_paths):
        learning_curves = []
        candidate_selections = []
        for i, (train_index, test_index) in enumerate(kfold):
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            active_learner = ActiveLearner(classifier=classifier,
                                           heuristic=heuristic,
                                           initial_n=initial_n,
                                           training_size=training_size,
                                           sample_size=sample_size,
                                           verbose=verbose,
                                           committee=committee,
                                           committee_samples=committee_samples,
                                           pool_n=pool_n,
                                           C=C,
                                           random_state=i,
                                           pool_random_state=i)
            active_learner.fit(X_train, y_train, X_test, y_test)        
            learning_curves.append(active_learner.learning_curve_)
            candidate_selections.append(active_learner.candidate_selections)
            print(i, end='')

        with open(pickle_path, 'wb') as f:
            pickle.dump((learning_curves, candidate_selections), f, protocol=4) 


def run_bandit_expt(X, y, kfold, classifier, committee, heuristics, pickle_paths,
    initial_n=50, training_size=300, sample_size=300, verbose=True,
    committee_samples=300, pool_n=300, C=1):
    """ Run Bandit experiment.
    """

    learning_curves = []
    heuristic_selections = []
    mus = []
    sigmas = []
    candidate_selections = []

    for i, (train_index, test_index) in enumerate(kfold):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        active_bandit = ActiveBandit(classifier=classifier,
                                       heuristics=heuristics,
                                       initial_n=initial_n,
                                       training_size=training_size,
                                       sample_size=sample_size,
                                       committee=committee,
                                       committee_samples=committee_samples,
                                       verbose=verbose,
                                       pool_n=pool_n,
                                       C=C)

        active_bandit.fit(X_train, y_train, X_test, y_test)
        
        learning_curves.append(active_bandit.learning_curve_)
        heuristic_selections.append(active_bandit.heuristic_selection)
        mus.append(active_bandit.all_prior_mus)
        sigmas.append(active_bandit.all_prior_sigmas)
        candidate_selections.append(active_bandit.candidate_selections)

        print(i, end='')

    outputs = [learning_curves, heuristic_selections, mus, sigmas, candidate_selections]
    for pickle_path, output in zip(pickle_paths, outputs):
        with open(pickle_path, 'wb') as f:
            pickle.dump(output, f, protocol=4)