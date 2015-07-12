""" The main routine of all active learning algorithms. """

import numpy as np
from sklearn import metrics
from mclearn.performance import balanced_accuracy_expected


def active_learn(training_pool, testing_pool, training_oracle, testing_oracle, total_n, initial_n,
                    random_n, active_learning_heuristic, classifier, compute_accuracy, n_classes,
                    committee=None, bag_size=None, verbose=False):
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
        
        n_classes : int
            The number of classes.
        
        committee : list of Classifier object
            A list that contains the committee of classifiers used by the query by bagging heuristics.
        
        bag_size : int
            The number of training examples used by each member in the committee.
        
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
            X_train_candidates, X_train=X_train, y_train=y_train, n_classes=n_classes,
            classifier=classifier, committee=committee, bag_size=bag_size)

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


def compute_accuracy(classifier, testing_pool, testing_oracle):
    """ Compute the accuracy of a classifier based on some test set. """
    
    y_pred = classifier.predict(testing_pool)
    confusion_test = metrics.confusion_matrix(testing_oracle, y_pred)
    return balanced_accuracy_expected(confusion_test)
