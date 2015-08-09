""" Various measures that evaluate the performance of a classifier. """

import numpy as np
from scipy.stats import beta
from scipy.integrate import trapz
from scipy.optimize import brentq
from pandas import DataFrame
from sklearn import metrics


def naive_accuracy(confusion):
    """ Compute the naive accuracy rate.
        
        Parameters
        ----------
        confusion : array, shape = [n_classes, n_classes]
            Where entry c_{ij} is the number of observations in class i but
            are classified as class j.
        
        Returns
        -------
        naive_accuracy : float
    """
    
    return p.trace(confusion) / np.sum(confusion)
    

def get_beta_parameters(confusion):
    """ Extract the beta parameters from a confusion matrix.
    
        Parameters
        ----------
        confusion : array, shape = [n_classes, n_classes]
            Where entry c_{ij} is the number of observations in class i but
            are classified as class j.
        
        Returns
        -------
        parameters: array of tuples
            Each tuple (alpha_i, beta_i) is the parameters of a Beta distribution
            that corresponds to class i.
    """

    alphas, betas = [], []
    
    # number of classes
    k = len(confusion)
    
    for i in range(k):
        # alpha is 1 plus the number of objects that are correctly classified
        alphas.append(1 + confusion[i, i])
        
        # beta is 1 plus the number of objects that are incorrectly classified
        betas.append(1 + confusion.sum(axis=1)[i] - confusion[i, i])
    
    return list(zip(alphas, betas))
    

def convolve_betas(parameters, res=0.001):
    """ Convolves k Beta distributions.
    
        Parameters
        ----------
        parameters : array of tuples
            Each tuple (alpha_i, beta_i) is the parameters of a Beta distribution.
        
        res : float, optional (default=0.001)
            The precision of the resulting convolution, measured as step size in
            the support.
        
        Returns
        -------
        convolution : array, shape = [k / res]
            The resulting convultion of the k Beta distributions, given the
            specified presicion `res`.
    """
    
    # number of convolution
    k = len(parameters)
    
    # sum of three probabilities ranges from 0 to k
    x = np.arange(0, k+res, res)
    
    # compute the individual beta pdfs
    pdfs = []
    for par in parameters:
        pdfs.append(beta.pdf(x, par[0], par[1]))
        
    # convolve k times
    convolution = pdfs[0]
    for i in range(1, k):
        convolution = np.convolve(convolution, pdfs[i])
        
    # reduce to the [0, k] support
    convolution = convolution[0:len(x)]
    
    # normalise so that all values sum to (1 / res)
    convolution = convolution / (sum(convolution) * res)
    
    return convolution
    
    
def balanced_accuracy_expected(confusion):
    """ Compute the expected value of the posterior balanced accuracy.
    
        Parameters
        ----------
        confusion : array, shape = [n_classes, n_classes]
            Where entry c_{ij} is the number of observations in class i but
            are classified as class j.
        
        Returns
        -------
        bal_accuracy_expected: float
    """
    
    # number of classes
    k = len(confusion)
    
    # extract beta distribution parameters from the confusion matrix 
    parameters = get_beta_parameters(confusion)
    
    # convolve the distributions and compute the expected value
    k = len(confusion)
    res = 0.001
    x = np.arange(0, k + res, res)
    bal_accuracy = convolve_betas(parameters, res)
    bal_accuracy_expected = (1/k) * np.dot(x, bal_accuracy * res)
    
    return bal_accuracy_expected
    

def beta_sum_pdf(x, parameters, res=0.001):
    """ Compute the pdf of the sum of beta distributions.
    
        Parameters
        ----------
        x : array
            A subset of the domain where we want evaluate the pdf.
            
        parameters : array of tuples
            Each tuple (alpha_i, beta_i) is the parameters of a Beta distribution.
        
        res : float, optional (default=0.001)
            The precision of the convolution, measured as step size in
            the support.
        
        Returns
        -------
        y : array
            The pdf evaulated at x.
    """
    
    convolution = convolve_betas(parameters, res)
    
    # convert x into a numpy array if it's not already
    x = np.array(x)
    
    # initialise the y vector
    y = np.array([np.nan] * len(x))
    
    # upper bound of support
    k = len(parameters)
    
    # set y to 0 if we're outside support
    y[(x < 0) | (x > k)] = 0
    
    # index in convolution vector that is closest to x
    c_index = np.int_(x / res)
    
    # fill in y vector
    y[np.isnan(y)] = convolution[c_index[np.isnan(y)]]
    
    return y

    
def beta_avg_pdf(x, parameters, res=0.001):
    """ Compute the pdf of the average of the k beta distributions.
    
        Parameters
        ----------
        x : array
            A subset of the domain where we want evaluate the pdf.
            
        parameters : array of tuples
            Each tuple (alpha_i, beta_i) is the parameters of a Beta distribution.
        
        res : float, optional (default=0.001)
            The precision of the convolution, measured as step size in
            the support.
        
        Returns
        -------
        y : array
            The pdf evaulated at x.
    """
    
    k = len(parameters)
    y = beta_sum_pdf(k * np.array(x), parameters, res)
    y = y * k
    
    return y
    
    
def beta_sum_cdf(x, parameters, res=0.001):
    """ Compute the cdf of the sum of the k beta distributions.
    
        Parameters
        ----------
        x : array
            A subset of the domain where we want evaluate the cdf.
            
        parameters : array of tuples
            Each tuple (alpha_i, beta_i) is the parameters of a Beta distribution.
        
        res : float, optional (default=0.001)
            The precision of the convolution, measured as step size in
            the support.
        
        Returns
        -------
        y : array
            The cdf evaulated at x.
    """
    
    convolution = convolve_betas(parameters, res)
    
    y = np.array([np.nan] * len(x))
    for i in range(len(x)):
        c_index = int(round(x[i] / res))
        if c_index <= 0:
            y[i] = 0
        elif c_index >= len(convolution):
            y[i] = 1
        else:
            y[i] = trapz(convolution[:c_index+1], dx=res)
    
    return y
    
    
def beta_avg_cdf(x, parameters, res=0.001):
    """ Compute the cdf of the average of the k beta distributions.
    
        Parameters
        ----------
        x : array
            A subset of the domain where we want evaluate the cdf.
            
        parameters : array of tuples
            Each tuple (alpha_i, beta_i) is the parameters of a Beta distribution.
        
        res : float, optional (default=0.001)
            The precision of the convolution, measured as step size in
            the support.
        
        Returns
        -------
        y : array
            The cdf evaulated at x.
    """
    
    x = np.array(x)
    k = len(parameters)
    y = beta_sum_cdf(k * x, parameters, res)
    
    return y
    

def beta_avg_inv_cdf(y, parameters, res=0.001):
    """ Compute the inverse cdf of the average of the k beta distributions.
    
        Parameters
        ----------
        y : float
            A float between 0 and 1 (the range of the cdf)
            
        parameters : array of tuples
            Each tuple (alpha_i, beta_i) is the parameters of a Beta distribution.
        
        res : float, optional (default=0.001)
            The precision of the convolution, measured as step size in
            the support.
        
        Returns
        -------
        x : float
            the inverse cdf of y
    """
    
    return brentq(lambda x: beta_avg_cdf([x], parameters, res)[0] - y, 0, 1)
    
 
def recall(confusion):
    """ Compute the recall from a confusion matrix.
        
        Parameters
        ----------
        confusion : array, shape = [n_classes, n_classes]
            Where entry c_{ij} is the number of observations in class i but
            are classified as class j.
        
        Returns
        -------
        recalls : array
            A list of recalls, one for each class.
    """
    
    # number of classes
    k = len(confusion)

    # extract recall from confusion matrix
    recalls = []
    for i in range(k):
        recalls.append(confusion[i, i] / confusion.sum(axis=1)[i])

    return recalls
    
    
def precision(confusion, classes, classifiers):
    """ Compute the precision from a confusion matrix.
        
        Parameters
        ----------
        confusion : array, shape = [n_classes, n_classes]
            Where entry c_{ij} is the number of observations in class i but
            are classified as class j.
        
        Returns
        -------
        precisions : array
            A list of precisions, one for each class.
    """

    # number of classes
    k = len(confusion)

    # extract recall from confusion matrix
    precisions = []
    for i in range(k):
        precisions.append(confusion[i, i] / confusion.sum(axis=0)[i])

    return precisions




def compute_balanced_accuracy(classifier, testing_pool, testing_oracle):
    """ Compute the accuracy of a classifier based on some test set.

        Parameters
        ----------
        classifier : Classifier object
            A trained instance of the Classifier object.

        testing_pool : array
            The feature matrix of the test examples.

        testing_oracle : array
            The target vector of the test examples.

        Returns
        -------
        balanced_accuracy_expected : float
            The expected balanced accuracy rate on the test set.

    """
    
    y_pred = classifier.predict(testing_pool)
    confusion_test = metrics.confusion_matrix(testing_oracle, y_pred)
    return balanced_accuracy_expected(confusion_test)
