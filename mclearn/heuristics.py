""" Heuristics used to query the most uncertain candidate out of the unlabelled pool. """

import numpy as np
import copy
from numpy.random import permutation
from sklearn.preprocessing import LabelBinarizer


def random_h(X_train, y_train, candidate_mask, classifier, n_candidates, train_mask, **kwargs):
    """ Return a random candidate.

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

        Returns
        -------
        best_candidates : int
            The indices of the best candidates (here it is random).
    """
    
    candidate_index = np.where(candidate_mask)[0]
    random_index = np.random.choice(candidate_index, n_candidates, replace=False)
    return random_index


def entropy_h(X_train, y_train, candidate_mask, classifier, n_candidates, train_mask, **kwargs):
    """ Return the candidate whose prediction vector displays the greatest Shannon entropy.

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

        Returns
        -------
        best_candidates : int
            The indices of the best candidates.

    """
    
    # predict probabilities
    probs = classifier.predict_proba(X_train[candidate_mask])
    
    # comptue Shannon entropy
    candidate_shannon = -np.sum(probs * np.log(probs), axis=1)

    # index the results properly
    shannon = np.empty(len(candidate_mask))
    shannon[:] = -np.inf
    shannon[candidate_mask] = candidate_shannon
    
    # pick the candidate with the greatest Shannon entropy
    best_candidates = np.argsort(-shannon)[:n_candidates]
    return best_candidates


def margin_h(X_train, y_train, candidate_mask, classifier, n_candidates, train_mask, **kwargs):
    """ Return the candidate with the smallest margin.
    
        The margin is defined as the difference between the two largest values
        in the prediction vector.

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

        Returns
        -------
        best_candidates : int
            The indices of the best candidates.
    """
    
    # predict probabilities
    probs = classifier.predict_proba(X_train[candidate_mask])
    
    # sort the probabilities from smallest to largest
    probs = np.sort(probs, axis=1)
    
    # compute the margin (difference between two largest values)
    candidate_margin = np.abs(probs[:,-1] - probs[:,-2])

    # index the results properly
    margin = np.empty(len(candidate_mask))
    margin[:] = +np.inf
    margin[candidate_mask] = candidate_margin
    
    # pick the candidate with the smallest margin
    best_candidates = np.argsort(margin)[:n_candidates]
    return best_candidates


def qbb_margin_h(X_train, y_train, candidate_mask, classifier, n_candidates, train_mask,
                 committee, **kwargs):
    """ Return the candidate with the smallest average margin.
    
        We first use bagging to train k classifiers. The margin is then defined as
        the average difference between the two largest values in the prediction vector.

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

        train_mask : boolean array
                The boolean array that tells us which data points are currently in the training set.

        committee : BaggingClassifier object
            The committee should have the same interface as scikit-learn BaggingClassifier.

        Returns
        -------
        best_candidates : int
            The indices of the best candidates.
    """
    
    # check that the max bagging sample is not too big
    committee.max_sample = min(committee.max_sample, len(y_candidates))

    # train and predict
    committee.fit(X_train[train_mask], y_train[train_mask])

    # predict
    n_samples = len(X_train[candidate_mask])
    n_classes = len(committee.classes_)
    probs = np.zeros((n_samples, n_classes))
    
    for member in committee.estimators_:
        memeber_prob = member.predict_proba(X_train[candidate_mask])

        if n_classes == len(member.classes_):
            probs += memeber_prob

        else:
            probs[:, member.classes_] += memeber_prob[:, range(len(member.classes_))]

    # average out the probabilities
    probs /= len(committee.estimators_)
    
    # sort the probabilities from smallest to largest
    probs = np.sort(probs, axis=1)
    
    # compute the margin (difference between two largest values)
    candidate_margin = np.abs(probs[:,-1] - probs[:,-2])

    # index the results properly
    margin = np.empty(len(candidate_mask))
    margin[:] = +np.inf
    margin[candidate_mask] = candidate_margin
    
    # pick the candidate with the smallest margin
    best_candidates = np.argsort(margin)[:n_candidates]
    return best_candidates


def qbb_kl_h(X_train, y_train, candidate_mask, classifier, n_candidates, train_mask,
             committee, **kwargs):
    """ Return the candidate with the largest average KL divergence from the mean.
    
        We first use bagging to train k classifiers. We then choose the candidate
        that has the largest Kullback–Leibler divergence from the average.

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

        train_mask : boolean array
                The boolean array that tells us which data points are currently in the training set.

        committee : BaggingClassifier object
            The committee should have the same interface as scikit-learn BaggingClassifier.

        Returns
        -------
        best_candidates : int
            The indices of the best candidates.
    """

    # check that the max bagging sample is not too big
    committee.max_sample = min(committee.max_sample, len(y_candidates))

    # train the committee
    committee.fit(X_train[train_mask], y_train[train_mask])

    # predict
    n_samples = len(X_train[candidate_mask])
    n_classes = len(committee.classes_)
    avg_probs = np.zeros((n_samples, n_classes))
    prob_list = []
    
    for member in committee.estimators_:
        memeber_prob = member.predict_proba(X_train[candidate_mask])

        if n_classes == len(member.classes_):
            avg_probs += memeber_prob
            prob_list.append(prob)

        else:
            avg_probs[:, member.classes_] += memeber_prob[:, range(len(member.classes_))]
            prob_list.append(prob)

    # average out the probabilities
    avg_probs /= len(committee.estimators_)

    # compute the KL divergence
    avg_kl = np.zeros(avg_probs.shape[0])
    for p in probs:
        member_kl = np.sum(p * np.log(p / avg_probs), axis=1)
        avg_kl += member_kl
    
    # average out the KL divergence
    avg_kl /= len(committee)

    # index the results properly
    kl = np.empty(len(candidate_mask))
    kl[:] = -np.inf
    kl[candidate_mask] = avg_kl
    
    # pick the candidate with the smallest margin
    best_candidates = np.argsort(-kl)[:n_candidates]
    return best_candidates





def compute_A(X, pi, classes):
    """ Compute the A matrix in the variance estimation technique.

        Parameters
        ----------
        X : array
            The feature matrix.

        pi : array
            The probability matrix predicted by the classifier.

        classes : array
            The list of class names ordered lexicographically.

        Returns
        -------
        A : array
            The A matrix as part of the variance calcucation.
    """
    
    n_classes = len(classes)
    n_features = X.shape[1]
    n_samples = X.shape[0]
    width = n_classes * n_features
    one_in_k = LabelBinarizer(pos_label=1, neg_label=0).fit_transform(classes)
    
    I_same = one_in_k.repeat(n_features, axis=0)
    I_same = np.tile(I_same, n_samples)
    
    I_diff = 1 - I_same
    
    A = np.tile(pi.flatten(), (width, 1))
    B = 1 - A
    C = -A
    D = pi.transpose().repeat(n_features, axis=0).repeat(n_classes, axis=1)
    E = X.transpose().repeat(n_classes, axis=1)
    E = np.tile(E, (n_classes, 1))
    G = A * B * I_same  + C * D * I_diff
    G = E * G
    outer = np.dot(G, G.transpose())
    
    return outer


def compute_F(X, pi, classes, C=1):
    """ Compute the F matrix in the variance estimation technqiue.

        Parameters
        ----------
        X : array
            The feature matrix.

        pi : array
            The probability matrix predicted by the classifier.

        classes : array
            The list of class names ordered lexicographically.

        C : float
            The regularisation parameter in logistic regression.

        Returns
        -------
        F : array
            The F matrix as part of the variance calcucation.
    """
    
    n_classes = len(classes)
    n_features = X.shape[1]
    n_samples = X.shape[0]
    width = n_classes * n_features
    
    I_diag = np.eye(width)
    
    mini_off_diag = 1 - np.eye(n_features)
    mini_zeros = np.zeros((n_features, n_features * n_classes))
    I_mini_off_diag = np.hstack((mini_off_diag, mini_zeros))
    I_mini_off_diag = np.tile(I_mini_off_diag, n_classes - 1)
    I_mini_off_diag = np.hstack((I_mini_off_diag, mini_off_diag))
    I_mini_off_diag = np.hsplit(I_mini_off_diag, n_classes)
    I_mini_off_diag = np.vstack(I_mini_off_diag)
    
    I_main_off_diag = 1 - I_diag - I_mini_off_diag
    
    M = np.tile(X.transpose(), (n_classes, 1))
    N = pi.transpose().repeat(n_features, axis=0)
    
    F_1 = np.dot(M * N * (1 - N), M.transpose()) + C
    F_2 = np.dot(M * N * (1 - N), M.transpose())
    F_3 = np.dot(M * N, M.transpose() * N.transpose())
    
    F = F_1 * I_diag + F_2 * I_mini_off_diag + F_3 * I_main_off_diag
    F = F / n_samples
    
    return F


def compute_pool_variance(X, pi, classes, C=1):
    """ Estimate the variance of the pool.

        Parameters
        ----------
        X : array
            The feature matrix.

        pi : array
            The probability matrix predicted by the classifier.

        classes : array
            The list of class names ordered lexicographically.

        C : float
            The regularisation parameter in logistic regression.

        Returns
        -------
        variance : float
            The estimated variance on the pool X.
    """

    A = compute_A(X, pi, classes)
    F = compute_F(X, pi, classes, C=C)
    return np.trace(np.dot(A, np.linalg.inv(F)))



def pool_variance_h(X_training_candidates, **kwargs):
    """ Return the candidate that will minimise the expected variance of the predictions.

        Parameters
        ----------
        X_training_candidates : array
            The feature matrix of the potential training candidates.

        C : float
            The regularisation parameter of Logistic Regression.

        pool_sample_size : int
            The size of the sample which will be used to estimate the variance/entropy.

        Returns
        -------
        best_candidate : int
            The index of the best candidate.
    """
    
    # extract parameters
    classifier = kwargs['classifier']
    X_train = kwargs['X_train'].copy()
    y_train = kwargs['y_train'].copy()
    classes = kwargs['classes']
    pool_sample_size = kwargs['pool_sample_size']
    C = kwargs['C']
    n_candidates = X_training_candidates.shape[0]
    n_features = X_training_candidates.shape[1]
    n_classes = len(classes)
    sigmas = np.zeros(n_candidates)

    # add an extra dummy example to the training set
    dummy_feature = np.zeros(n_features)
    X_train = np.vstack((X_train, dummy_feature))
    y_train = np.concatenate((y_train, ['None']))

    # predict probabilities
    probs = classifier.predict_proba(X_training_candidates)

    # construct the sample pool
    sample_pool_index = permutation(n_candidates)[:pool_sample_size]
    sample_pool = X_training_candidates[sample_pool_index]

    for i in np.arange(n_candidates):
        candidate_features = X_training_candidates[i]
        candidate_sigmas = np.zeros(n_classes)
        X_train[-1] = candidate_features

        # assume the candidate is in each of the classes
        for j in np.arange(n_classes):
            assumed_class = classes[j]
            y_train[-1] = assumed_class

            # re-train the classifier
            classifier.fit(X_train, y_train)
            pi = classifier.predict_proba(sample_pool)
            candidate_sigmas[j] = compute_pool_variance(sample_pool, pi, classes, C=C)

        sigmas[i] = np.dot(probs[i], candidate_sigmas)

    best_candidate = np.argmin(sigmas)
    return [best_candidate]


def compute_pool_entropy(pi):
    """ Estimate the variance of the pool.

        Parameters
        ----------
        pi : array
            The probability matrix predicted by the classifier.

        Returns
        -------
        entropy : float
            The estimated entropy on the pool.
    """

    return -np.sum(pi * np.log(pi))


def pool_entropy_h(X_training_candidates, **kwargs):
    """ Return the candidate that will minimise the expected entropy of the predictions.

        Parameters
        ----------
        X_training_candidates : array
            The feature matrix of the potential training candidates.

        classes : int
            The name of classes.

        Returns
        -------
        best_candidate : int
            The index of the best candidate.
    """
    
    # extract parameters
    classifier = kwargs['classifier']
    X_train = kwargs['X_train'].copy()
    y_train = kwargs['y_train'].copy()
    classes = kwargs['classes']
    pool_sample_size = kwargs['pool_sample_size']
    n_candidates = X_training_candidates.shape[0]
    n_features = X_training_candidates.shape[1]
    n_classes = len(classes)
    entropy = np.zeros(n_candidates)

    # add an extra dummy example to the training set
    dummy_feature = np.zeros(n_features)
    X_train = np.vstack((X_train, dummy_feature))
    y_train = np.concatenate((y_train, ['None']))

    # predict probabilities
    probs = classifier.predict_proba(X_training_candidates)

    # construct the sample pool
    sample_pool_index = permutation(n_candidates)[:pool_sample_size]
    sample_pool = X_training_candidates[sample_pool_index]

    for i in np.arange(n_candidates):
        candidate_features = X_training_candidates[i]
        candidate_entropy = np.zeros(n_classes)
        X_train[-1] = candidate_features

        # assume the candidate is in each of the classes
        for j in np.arange(n_classes):
            assumed_class = classes[j]
            y_train[-1] = assumed_class

            # re-train the classifier
            classifier.fit(X_train, y_train)
            pi = classifier.predict_proba(sample_pool)
            candidate_entropy[j] = compute_pool_entropy(pi)

        entropy[i] = np.dot(probs[i], candidate_entropy)

    best_candidate = np.argmin(entropy)
    return [best_candidate]