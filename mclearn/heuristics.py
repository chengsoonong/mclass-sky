""" Heuristics used to query the most uncertain candidate out of the unlabelled pool. """

import numpy as np
from numpy.random import permutation
from sklearn.preprocessing import LabelBinarizer

def random_h(X_training_candidates, **kwargs):
    """ Return a random candidate.

        Parameters
        ----------
        X_training_candidates : array
            The feature matrix of the potential training candidates.

        Returns
        -------
        best_candidate : int
            The index of the best candidate (here it is random).

    """
    
    random_index = np.random.choice(np.arange(0, len(X_training_candidates)), 1, replace=False)
    return random_index


def entropy_h(X_training_candidates, **kwargs):
    """ Return the candidate whose prediction vector displays the greatest Shannon entropy.

        Parameters
        ----------
        X_training_candidates : array
            The feature matrix of the potential training candidates.

        Returns
        -------
        best_candidate : int
            The index of the best candidate.

    """
    
    # get the classifier
    classifier = kwargs['classifier']
    
    # predict probabilities
    probs = classifier.predict_proba(X_training_candidates)
    
    # comptue Shannon entropy
    shannon = -np.sum(probs * np.log(probs), axis=1)
    
    # pick the candidate with the greatest Shannon entropy
    greatest_shannon = np.argmax(shannon)
    
    return [greatest_shannon]


def margin_h(X_training_candidates, **kwargs):
    """ Return the candidate with the smallest margin.
    
        The margin is defined as the difference between the two largest values
        in the prediction vector.

        Parameters
        ----------
        X_training_candidates : array
            The feature matrix of the potential training candidates.

        Returns
        -------
        best_candidate : int
            The index of the best candidate.
    """
    
    # get the classifier
    classifier = kwargs['classifier']
    
    # predict probabilities
    probs = classifier.predict_proba(X_training_candidates)
    
    # sort the probabilities from smallest to largest
    probs = np.sort(probs, axis=1)
    
    # compute the margin (difference between two largest values)
    margins = np.abs(probs[:,-1] - probs[:,-2])
    
    # pick the candidate with the smallest margin
    smallest_margin = np.argmin(margins)
    
    return [smallest_margin]


def qbb_margin_h(X_training_candidates, **kwargs):
    """ Return the candidate with the smallest average margin.
    
        We first use bagging to train k classifiers. The margin is then defined as
        the average difference between the two largest values in the prediction vector.

        Parameters
        ----------
        X_training_candidates : array
            The feature matrix of the potential training candidates.

        Returns
        -------
        best_candidate : int
            The index of the best candidate.
    """
    
    # extract parameters
    committee = kwargs['committee']
    bag_size = kwargs['bag_size']
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    classes = kwargs['classes']
    n_classes = len(classes)
    
    # intialise probability matrix
    probs = np.zeros((len(X_training_candidates), n_classes))
    
    # train each member of the committee
    for member in committee:
        
        # randomly select a bag of samples
        member_train_index = np.random.choice(np.arange(0, len(y_train)), bag_size, replace=True)
        member_X_train = X_train[member_train_index]
        member_y_train = y_train[member_train_index]
        
        # train member and predict
        member.fit(member_X_train, member_y_train)
        prob = member.predict_proba(X_training_candidates)
        
        # make sure all class predictions are present
        prob_full = np.zeros((prob.shape[0], n_classes))
        
        # case 1: only galaxy predictions are generated
        if prob.shape[1] == 1:
            prob_full[:,0] += prob
            
        # case 2: only galaxy and star predictions are generated
        if prob.shape[1] == 2:
            prob_full[:,0] += prob[:,0]
            prob_full[:,2] += prob[:,1]
            
        # case 3: all class predictions are generated
        else:
            prob_full += prob
            
        # accumulate probabilities
        probs += prob_full
            
    # average out the probabilities
    probs /= len(committee)
    
    # sort the probabilities from smallest to largest
    probs = np.sort(probs, axis=1)
    
    # compute the margin (difference between two largest values)
    margins = np.abs(probs[:,-1] - probs[:,-2])

    # pick the candidate with the smallest margin
    smallest_margin = np.argmin(margins)
    
    return [smallest_margin]


def qbb_kl_h(X_training_candidates, **kwargs):
    """ Return the candidate with the largest average KL divergence from the mean.
    
        We first use bagging to train k classifiers. We then choose the candidate
        that has the largest Kullback–Leibler divergence from the average.

        Parameters
        ----------
        X_training_candidates : array
            The feature matrix of the potential training candidates.

        Returns
        -------
        best_candidate : int
            The index of the best candidate.
    """
    
    # extract parameters
    committee = kwargs['committee']
    bag_size = kwargs['bag_size']
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    classes = kwargs['classes']
    n_classes = len(classes)
    
    # intialise probability matrix
    avg_probs = np.zeros((len(X_training_candidates), n_classes))
    probs = []
    
    # train each member of the committee
    for member in committee:
        
        # randomly select a bag of samples
        member_train_index = np.random.choice(np.arange(0, len(y_train)), bag_size, replace=True)
        member_X_train = X_train[member_train_index]
        member_y_train = y_train[member_train_index]
        
        # train member and predict
        member.fit(member_X_train, member_y_train)
        prob = member.predict_proba(X_training_candidates)
        
        # make sure all class predictions are present
        prob_full = np.zeros((prob.shape[0], n_classes))
        
        # case 1: only galaxy predictions are generated
        if prob.shape[1] == 1:
            prob_full[:,0] += prob
            
        # case 2: only galaxy and star predictions are generated
        if prob.shape[1] == 2:
            prob_full[:,0] += prob[:,0]
            prob_full[:,2] += prob[:,1]
            
        # case 3: all class predictions are generated
        else:
            prob_full += prob
            
        # accumulate probabilities
        probs.append(prob_full)
        avg_probs += probs[-1]
        
    # average out the probabilities
    avg_probs /= len(committee)
    
    # compute the KL divergence
    avg_kl = np.zeros(avg_probs.shape[0])
    for p in probs:
        kl = np.sum(p * np.log(p / avg_probs), axis=1)
        avg_kl += kl
    
    # average out the KL divergence
    avg_kl /= len(committee)
    
    # extract the candidate with the largest average divergence
    largest_kl = np.argmax(avg_kl)
    
    return [largest_kl]



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