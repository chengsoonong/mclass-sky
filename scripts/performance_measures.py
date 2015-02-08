""" Classifier Performance

"""

import numpy as np
from scipy.stats import beta
from scipy.integrate import trapz
from scipy.optimize import brentq
from pandas import DataFrame


def naive_accuracy(confusions, classifiers):
    """
    Input: confusion matrix and array of classifier names (column index)
    """
    
    results_dict = {}
    
    for clf, conf in confusions.items():
        results_dict[clf] = np.trace(conf) / np.sum(conf)

    results_df = DataFrame(results_dict, index=["Accuray"])
    results_df = results_df.reindex(columns=classifiers, copy=False)
    
    return results_df
    

def get_beta_parameters(confusion):
    alphas, betas = [], []
    
    # number of classes
    k = len(confusions[name])
    
    for i in range(k):
        # alpha is 1 plus the number of objects that are correctly classified
        alphas.append(1 + confusions[name][i, i])
        
        # beta is 1 plus the number of objects that are incorrectly classified
        betas.append(1 + confusions[name].sum(axis=1)[i] - confusions[name][i, i])
        
    parameters = list(zip(alphas, betas))
    
    return parameters
    
    

def convolve_betas(parameters, res=0.001):
    """ Convolves k Beta distributions. Parameters is a list of tuples (alpha_i, beta_i)"""
    
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
    """ Compute the expected value of the posterior balanced accuracy. Input is a numpy matrix"""
    
    alphas, betas = [], []
    k = len(confusion) # number of classes
    
    for i in range(k):
        # alpha is 1 plus the number of objects that are correctly classified
        alphas.append(1 + confusion[i, i])
        
        # beta is 1 plus the number of objects that are incorrectly classified
        betas.append(1 + confusion.sum(axis=1)[i] - confusion[i, i])
    
    parameters = list(zip(alphas, betas))
    
    # convolve the distributions and compute the expected value
    res = 0.001
    x = np.arange(0, k + res, res)
    bal_accuracy = convolve_betas(parameters, res)
    bal_accuracy_expected = (1/k) * np.dot(x, bal_accuracy * res)
    
    return bal_accuracy_expected
    
    

def beta_sum_pdf(x, parameters, res=0.001):
    """ input x is an array """
    
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
    """ input x is an array """
    
    k = len(parameters)
    y = beta_sum_pdf(k * np.array(x), parameters, res)
    y = y * k
    
    return y
    
    
    
def beta_sum_cdf(x, parameters, res=0.001):
    """ input x is an array """
    
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
    """ input x is an array """
    x = np.array(x)
    k = len(parameters)
    y = beta_sum_cdf(k * x, parameters, res)
    
    return y
    

def beta_avg_inv_cdf(y, parameters, res=0.001):
    return brentq(lambda x: beta_avg_cdf([x], parameters, res)[0] - y, 0, 1)
    
 
def recall(confusion, classes, classifiers):
    """
    Input: confusion matrix
           array of class names (row index)
           array of classifier names (column index)
    """

    # initialise dict to store results
    results_dict = {c: {} for c in classes}
    
    # extract recall from confusion matrix
    for clf, conf in confusions.items():
        for i in range(len(classes)):
            results_dict[classes[i]] = conf[i, i] / conf.sum(axis=1)[i]

    # organise as DataFrame
    results_df = DataFrame.from_dict(results_dict, orient="index")
    results_df = results.df.reindex(columns=classifiers, copy=False)
    
    return results_df
    
    
    
def precision(confusion, classes, classifiers):
    """
    Input: confusion matrix
           array of class names (row index)
           array of classifier names (column index)
    """

    # initialise dict to store results
    results_dict = {c: {} for c in classes}
    
    # extract recall from confusion matrix
    for clf, conf in confusions.items():
        for i in range(len(classes)):
            results_dict[classes[i]] = conf[i, i] / conf.sum(axis=0)[i]

    # organise as DataFrame
    results_df = DataFrame.from_dict(results_dict, orient="index")
    results_df = results.df.reindex(columns=classifiers, copy=False)
    
    return results_df