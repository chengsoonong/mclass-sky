import numpy as np
import scipy as sp
from rank_nullspace import nullspace
from sklearn.preprocessing import KernelCenterer


def calculate_knfst(K, labels):
    '''
    Calculates projection matrix of KNFST
    '''
    classes = np.unique(labels)
    if len(classes) < 2:
        raise Exception("KNFST requires 2 or more classes")
    n, m = K.shape
    if n != m:
        raise Exception("Kernel matrix must be quadratic")
    
    centered_k = KernelCenterer().fit_transform(K)
    basis_values, basis_vecs = np.linalg.eig(centered_k)

    idx = basis_values.argsort()[::-1]
    basis_values = basis_values[idx]
    basis_vecs = basis_vecs[:, idx]

    basis_vecs = basis_vecs[:,basis_values > 1e-12]
    basis_values = basis_values[basis_values > 1e-12]
 
    basis_values = np.diag(1.0/np.sqrt(basis_values))
    basis_vecs  = basis_vecs.dot(basis_values)

    L = np.zeros([n,n])
    for cl in classes:
        for idx1, x in enumerate(labels == cl):
            for idx2, y in enumerate(labels == cl):
                if x and y:
                    L[idx1, idx2] = 1.0/np.sum(labels==cl)
    
    M = np.ones([m,m])/m
    H = ((np.eye(m,m)-M).dot(basis_vecs)).T.dot(K).dot(np.eye(n,m)-L)
    
    t_sw = H.dot(H.T)
    
    eigenvecs = nullspace(t_sw)
    if eigenvecs.shape[1] < 1:
        eigenvals, eigenvecs = np.linalg.eig(t_sw)
        eigenvals = np.diag(eigenvals)
        min_idx = eigenvals.argsort()[0]
        eigenvecs = eigenvecs[:, min_idx]
        
    proj = (np.eye(m,m)-M).dot(basis_vecs).dot(eigenvecs)
    return proj

