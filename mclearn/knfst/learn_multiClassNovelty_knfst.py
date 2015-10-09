import numpy as np
import scipy as sp
from knfst import calculate_knfst


def learn_multiclass_novelty_knfst(K, labels):
    '''
    Calculate multi-class KNFST model for multi-class novelty detection
    
    INPUT
      K: NxN kernel matrix containing similarities of n training samples
      labels: Nx1 column vector containing multi-class labels of N training samples

    OUTPUT
      proj: Projection of KNFST
      target_points: The projections of training data into the null space
    '''

    classes = np.unique(labels)
    proj = calculate_knfst(K, labels)
    target_points = []
    for cl in classes:
        k_cl = K[labels==cl]
        target_points.append(np.mean(k_cl.dot(proj), axis=0))
        
    return proj, target_points


