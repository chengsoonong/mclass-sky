import numpy as np
import active
import config

def compute_accuracy(w, X_testing, Y_testing):
    size = X_testing.shape[0]
    predictions = active.linear_predictor(X_testing, w)
    results = predictions == Y_testing
    correct = np.count_nonzero(results)
    accuracy = correct/size
    return accuracy
    
def weights_matrix(n, iterations, X_training, Y_training, center='ac', 
                   sample=1, M=None):
    testing = 3
    matrix_of_weights = []
    for i in range(n):
        weights = active.active(X_training, Y_training, iterations, center=center,
                                sample = sample, testing=testing, M=M)[2]
        matrix_of_weights.append(weights)
    return matrix_of_weights

def experiment(n, iterations, X_testing, Y_testing, X_training, Y_training,
               center='ac', sample = 1, M=None):
    config.reset()

    testing=3
    matrix_of_weights = weights_matrix(n, iterations, X_training, Y_training, 
                                       center=center, sample=sample, M=M)
    matrix_of_accuracies = []
    
    for weights in matrix_of_weights:
        accuracies = []
        for weight in weights:
            accuracy = compute_accuracy(weight, X_testing, Y_testing)
            accuracies.append(accuracy)
        matrix_of_accuracies.append(accuracies)
    
    matrix_of_accuracies = np.array(matrix_of_accuracies)
    sum_of_accuracies = matrix_of_accuracies.sum(axis=0)
    average_accuracies = sum_of_accuracies/n
    return average_accuracies