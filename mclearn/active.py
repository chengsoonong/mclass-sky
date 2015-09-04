""" The main routine of all active learning algorithms. """

import pickle
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle

from .heuristics import random_h
from .performance import compute_balanced_accuracy


class ActiveLearner(object):
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

    def __init__(self, classifier,
                 heuristic=random_h,
                 accuracy_fn=compute_balanced_accuracy,
                 initial_n=20,
                 training_size=100,
                 sample_size=20,
                 n_candidates=1,
                 verbose=False,
                 **kwargs):

        self.classifier = classifier
        self.heuristic = heuristic
        self.accuracy_fn = accuracy_fn
        self.initial_n = initial_n
        self.training_size = training_size
        self.current_training_size = 0
        self.n_candidates = n_candidates
        self.sample_size = sample_size
        self.verbose = verbose
        self.learning_curve_ = []
        self.kwargs = kwargs


    def best_candidates(self, X_train, y_train, candidate_mask, classifier, n_candidates, train_mask, **kwargs):
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

        return self.heuristic(X_train, y_train, candidate_mask, classifier, n_candidates, train_mask, **kwargs)


    def _random_sample(self, pool_size, train_mask, sample_size):
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
            self.learning_curve_.append(accuracy)

        # keep training the classifier until we have a desired sample size
        while np.sum(train_mask) < self.training_size:
            
            # select a random sample from the unlabelled pool
            candidate_mask = self._random_sample(pool_size, train_mask, self.sample_size)

            # pick the index of the best candidates
            best_candidates = self.best_candidates(X_train, y_train, candidate_mask,
                                                   self.classifier, self.n_candidates, train_mask,
                                                   **self.kwargs)

            # retrain the classifier
            train_mask[best_candidates] = True
            self.classifier.fit(X_train[train_mask], y_train[train_mask])
            self.current_training_size += len(best_candidates)

            # obtain the next data point of the learning curve
            if X_test is not None and y_test is not None:
                accuracy = self.accuracy_fn(self.classifier, X_test, y_test)
                self.learning_curve_.append(accuracy)

            # print progress after every 100 queries
            if self.verbose:
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







def run_active_learning_with_heuristic(heuristic, classifier,
    training_pool, testing_pool, training_oracle, testing_oracle, balanced_pool=False,
    full_sample_size=60000, n_trials=10, total_n=1000, initial_n=20, random_n=60000,
    committee=None, bag_size=10000, classes=['Galaxy', 'Star', 'Quasar'], C=None,
    pool_sample_size=300, pickle_path=None):
    """ Experiment routine with a partciular classifier heuristic.

        Parameters
        ----------
        heuristic : function
            This is the function that implements the active learning rule. Given a set
            of training candidates and the classifier as inputs, the function will
            return index array of candidate(s) with the highest score(s).

        classifier : Classifier object
            A classifier object that will be used to train and test the data.
            It should have the same interface as scikit-learn classifiers.

        training_pool : array, shape = [n_samples, n_features]
            The feature matrix of all the training examples. Throughout the training phase,
            the active learner will select an oject from this pool to query to oracle.
            
        testing_pool : array, shape = [n_samples, n_features]
            The feature matrix of the test examples, which will be used to assess the accuracy
            rate of the active learner.
            
        training_oracle : array, shape = [n_samples]
            The array of class labels corresponding to the training examples.
            
        testing_oracle : array, shape = [n_samples]
            The array of class labels corresponding to the test examples.

        balanced_pool : boolean
            Whether the class disribution in the training pool should be uniform.

        full_sample_size : int
            The size of the training pool in each trial of the experiment.

        n_trials : int
            The number trials the experiment will be run.

        total_n : int
            The total number of samples that the active learner will query.
            
        initial_n : int
            The number of samples that the active learner will randomly select at the beginning
            to get the algorithm started.
            
        random_n : int
            At each iteration, the active learner will pick a random of sample of examples.
            It will then compute a score for each of example and query the one with the
            highest score according to the active learning rule. If random_n is set to 0,
            the entire training pool will be sampled (which can be inefficient with large
            datasets).
        
        committee : list of Classifier object
            A list that contains the committee of classifiers used by the query by bagging heuristics.
        
        bag_size : int
            The number of training examples used by each member in the committee.

        classes : array
            The names of the targets.

        C : float
            The regularisation parameter of Logistic Regression.

        pickle_paths : array
            List of paths where the learning curves can be saved.

        Returns
        -------
        learning_curves : array
            If no pickle paths are provided, the learning curves will be returned.
    """
    
    sub_sample_size = full_sample_size // 3
    learning_curves = []

    if balanced_pool:
        i_max = sub_sample_size * n_trials
        i_step = sub_sample_size
    else:
        i_max = full_sample_size * n_trials
        i_step = full_sample_size

    for i in np.arange(0, i_max, i_step):
        if balanced_pool:
            is_galaxy = training_oracle == 'Galaxy'
            is_star = training_oracle == 'Star'
            is_quasar = training_oracle == 'Quasar'

            galaxy_features = training_pool[is_galaxy]
            star_features = training_pool[is_star]
            quasar_features = training_pool[is_quasar]

            training_galaxy = galaxy_features[i:i+sub_sample_size]
            training_star = star_features[i:i+sub_sample_size]
            training_quasar = quasar_features[i:i+sub_sample_size]
            
            training_sub_pool = np.concatenate((training_galaxy, training_star, training_quasar), axis=0)
            training_sub_oracle = np.concatenate((np.repeat('Galaxy', sub_sample_size),
                np.repeat('Star', sub_sample_size), np.repeat('Quasar', sub_sample_size)))
        else:
            training_sub_pool = training_pool[i:i+full_sample_size]
            training_sub_oracle = training_oracle[i:i+full_sample_size]
    
        # train the active learner
        active_learner = ActiveLearner(classifier=classifier,
                                       heuristic=heuristic,
                                       initial_n=initial_n,
                                       training_size=total_n,
                                       sample_size=random_n,
                                       verbose=True,
                                       committee=committee)
        active_learner.fit(training_pool, training_oracle, testing_pool, testing_oracle)        
        learning_curves.append(active_learner.learning_curve_)
    
    print('\n')
    
    if pickle_path:
        with open(pickle_path, 'wb') as f:
            pickle.dump(learning_curves, f, pickle.HIGHEST_PROTOCOL) 

    else:
        return learning_curves



def active_learning_experiment(data, feature_cols, target_col, classifier,
    heuristics, committee, pickle_paths, degree=1, n_trials=10, total_n=1000, balanced_pool=False,
    C=None, pool_sample_size=300, random_n=60000):
    """ Run an active learning experiment with specified heuristics.

        Parameters
        ----------
        data : DataFrame
            The DataFrame containing all the features and target.

        feature_cols : array
            The list of column names of the features.

        target_col: array
            The name of the target column in the DataFrame.

        classifier : Classifier object
            A classifier object that will be used to train and test the data.
            It should have the same interface as scikit-learn classifiers.

        heuristics : array
            The list of heuristics to be experimented on. Each heuristic is
            a function that implements the active learning rule. Given a set
            of training candidates and the classifier as inputs, the function will
            return index array of candidate(s) with the highest score(s).

        committee : list of Classifier objects
            A list that contains the committee of classifiers used by the query by bagging heuristics.

        pickle_paths : array
            List of paths where the learning curves can be saved.

        degree : int
            If greater than 1, the data will be transformed polynomially with the given degree.

        n_trials : int
            The number trials the experiment will be run.

        total_n : int
            The total number of samples that the active learner will query.

        balanced_pool : boolean
            Whether the class disribution in the training pool should be uniform.

        C : float
            The regularisation parameter of Logistic Regression.

        random_n : int
            At each iteration, the active learner will pick a random of sample of examples.
            It will then compute a score for each of example and query the one with the
            highest score according to the active learning rule. If random_n is set to 0,
            the entire training pool will be sampled (which can be inefficient with large
            datasets).

    """

    # 70/30 split of training and test sets
    training_pool, testing_pool, training_oracle, testing_oracle = train_test_split(
        np.array(data[feature_cols]), np.array(data[target_col]), train_size=0.7)

    # shuffle and randomise data
    training_pool, training_oracle = shuffle(training_pool, training_oracle, random_state=14)

    # do a polynomial transformation
    if degree > 1:
        poly_features = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
        training_pool = poly_features.fit_transform(training_pool)
        testing_pool = poly_features.transform(testing_pool)

    for heuristic, pickle_path in zip(heuristics, pickle_paths):
        run_active_learning_with_heuristic(heuristic, classifier, training_pool,
            testing_pool, training_oracle, testing_oracle, n_trials=n_trials, total_n=total_n,
            committee=committee, pickle_path=pickle_path, balanced_pool=balanced_pool, C=C,
            pool_sample_size=pool_sample_size, random_n=random_n)
