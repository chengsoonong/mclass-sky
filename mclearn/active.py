""" The main routine of all active learning algorithms. """

import mclearn
import pickle
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle


def active_learn(training_pool, testing_pool, training_oracle, testing_oracle, total_n, initial_n,
                    random_n, active_learning_heuristic, classifier, compute_accuracy, classes,
                    committee=None, bag_size=None, C=None, pool_sample_size=300, verbose=False):
    """ Conduct active learning and return a learning curve.
    
        Parameters
        ----------
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
            
        active_learning_heuristic : function
            This is the function that implements the active learning rule. Given a set
            of training candidates and the classifier as inputs, the function will
            return index array of candidate(s) with the highest score(s).
            
        classifier : Classifier object
            A classifier object that will be used to train and test the data.
            It should have the same interface as scikit-learn classifiers.
               
        compute_accuracy : function
            Given a trained classifier, a test set, and a test oracle, this function
            will return the accuracy rate.
        
        classes : int
            The name of classes.
        
        committee : list of Classifier object
            A list that contains the committee of classifiers used by the query by bagging heuristics.
        
        bag_size : int
            The number of training examples used by each member in the committee.

        C : float
            The regularisation parameter of Logistic Regression.

        pool_sample_size : int
            The size of the sample which will be used to estimate the variance/entropy.
        
        verbose : boolean
            If set to True, progress is printed to standard output after every 100 iterations.
            
        Returns
        -------
        learning_curve : array
            Every time the active learner queries the oracle, it will re-train the classifier
            and run it on the test data to get an accuracy rate. The learning curve is
            simply the array containing all of these accuracy rates.
    """
    
    n_features = training_pool.shape[1]
    learning_curve = []
    
    # the training examples that haven't been queried
    unlabelled_pool, unlabelled_oracle = training_pool.copy(), training_oracle.copy()
    
    # training examples that have been queried
    X_train = np.empty((0, n_features), float)
    y_train = np.array([])
    
    # select an initial random sample from the pool and train the classifier
    candidate_index = np.random.choice(np.arange(0, len(unlabelled_oracle)), initial_n, replace=False)
    
    # get the feature matrix and labels for our candidates
    X_train_candidates = unlabelled_pool[candidate_index]
    y_train_candidates = unlabelled_oracle[candidate_index]
    
    # add candidate to current training pool
    X_train = np.append(X_train, X_train_candidates, axis=0)
    y_train = np.concatenate((y_train, y_train_candidates))
                                  
    # remove candidate from existing unlabelled pool
    unlabelled_pool = np.delete(unlabelled_pool, candidate_index, axis=0)
    unlabelled_oracle = np.delete(unlabelled_oracle, candidate_index)
    
    # train and test the classifer
    classifier.fit(X_train, y_train)
    accuracy = compute_accuracy(classifier, testing_pool, testing_oracle)
    learning_curve.append(accuracy)

    
    while len(y_train) < total_n:
        
        # select a random sample from the unlabelled pool
        candindate_size = min(random_n, len(unlabelled_oracle))
        candidate_index = np.random.choice(np.arange(0, len(unlabelled_oracle)), candindate_size, replace=False)
        
        # get the feature matrix and labels for our candidates
        X_train_candidates = unlabelled_pool[candidate_index]
        y_train_candidates = unlabelled_oracle[candidate_index]

        # pick the best candidate using an active learning heuristic
        best_index = active_learning_heuristic(
            X_train_candidates, X_train=X_train, y_train=y_train, classes=classes,
            classifier=classifier, committee=committee, bag_size=bag_size, C=C,
            pool_sample_size=pool_sample_size)

        # add candidate to current training pool
        X_train = np.append(X_train, X_train_candidates[best_index], axis=0)
        y_train = np.concatenate((y_train, y_train_candidates[best_index]))

        # remove candidate from existing unlabelled pool
        best_index_in_unlabelled = candidate_index[best_index]
        unlabelled_pool = np.delete(unlabelled_pool, best_index_in_unlabelled, axis=0)
        unlabelled_oracle = np.delete(unlabelled_oracle, best_index_in_unlabelled)

        # train and test the classifer again
        classifier.fit(X_train, y_train)
        accuracy = compute_accuracy(classifier, testing_pool, testing_oracle)
        learning_curve.append(accuracy)
        
        # print progress after every 100 queries
        if verbose and len(y_train) % 100 == 0:
            if len(y_train) % 1000 == 0:
                print(len(y_train), end='')
            else:
                print('.', end='')
    
    return learning_curve




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
        learning_curve = active_learn(
            training_sub_pool, testing_pool, training_sub_oracle, testing_oracle,
            total_n=total_n, initial_n=initial_n, random_n=random_n,
            active_learning_heuristic=heuristic, classifier=classifier,
            compute_accuracy=mclearn.performance.compute_balanced_accuracy,
            classes=classes, committee=committee, bag_size=bag_size, C=C,
            pool_sample_size=pool_sample_size, verbose=True)

        learning_curves.append(learning_curve)
    
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
