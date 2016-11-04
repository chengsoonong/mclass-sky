import numpy as np
import accpm
import pandas as pd
import scipy.optimize as opt
from scipy.special import expit # The logistic sigmoid function 

def cost(w, X, y, c=0):
    """
    Returns the cross-entropy error function with (optional) sum-of-squares regularization term.
    
    w -- parameters
    X -- dataset of features where each row corresponds to a single sample
    y -- dataset of labels where each row corresponds to a single sample
    c -- regularization coefficient (default = 0)
    """
    outputs = expit(X.dot(w)) # Vector of outputs (or predictions)

    return -( y.transpose().dot(np.log(outputs)) + (1-y).transpose().dot(np.log(1-outputs)) ) + c*0.5*w.dot(w)

def grad(w, X, y, c=0):
    """
    Returns the gradient of the cross-entropy error function with (optional) sum-of-squares regularization term.
    """
    outputs = expit(X.dot(w))
    return X.transpose().dot(outputs-y) + c*w
    
def train(X, y,c=0):
    """
    Returns the vector of parameters which minimizes the error function via the BFGS algorithm.
    """
    initial_values = np.zeros(X.shape[1]) # Error occurs if inital_values is set too high
    return opt.minimize(cost, initial_values, jac=grad, args=(X,y,c), 
                        method='BFGS', options={'disp' : False, 'gtol' : 1e-03}).x

def predict(w, X):
    """
    Returns a vector of predictions. 
    """
    return expit(X.dot(w))

def compute_accuracy(predictions, y):
    """
    predictions -- dataset of predictions (or outputs) from a model
    y -- dataset of labels where each row corresponds to a single sample
    """
    predictions = predictions.round()
    size = predictions.shape[0]
    results = predictions == y
    correct = np.count_nonzero(results)
    accuracy = correct/size
    return accuracy

def compute_weights(X_training, Y_training, iterations):
    weights = []

    size = X_training.shape[0]
    index = np.arange(size)
    np.random.shuffle(index)

    for i in range(1, iterations + 1):
        X_i, Y_i =  X_training[index[:i]], Y_training[index[:i]]
        weight = train(X_i, Y_i)
        weights.append(weight)

    return weights

def weights_matrix(n, iterations, X_training, Y_training):
    matrix_of_weights = []
    np.random.seed(1)
    for i in range(n):
        weights = compute_weights(X_training, Y_training, iterations)
        matrix_of_weights.append(weights)
    return np.array(matrix_of_weights)

def experiment(n, iterations, X_testing, Y_testing, X_training, Y_training):

    matrix_of_weights = weights_matrix(n, iterations, X_training, Y_training)
    matrix_of_accuracies = []
    
    for weights in matrix_of_weights:
        accuracies = []
        for weight in weights:
            predictions = predict(weight, X_testing)
            accuracy = compute_accuracy(predictions, Y_testing)
            accuracies.append(accuracy)
        matrix_of_accuracies.append(accuracies)

    matrix_of_accuracies = np.array(matrix_of_accuracies)
    sum_of_accuracies = matrix_of_accuracies.sum(axis=0)
    average_accuracies = sum_of_accuracies/n
    return average_accuracies
