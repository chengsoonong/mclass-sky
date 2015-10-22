import numpy as np
import scipy as sp

def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return v[rank:].T.copy()


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
        
    centered_k = center_kernel_matrix(K)
    basis_values, basis_vecs = np.linalg.eig(centered_k)
  
    basis_vecs = basis_vecs[:,basis_values > 1e-12]
    basis_values = basis_values[basis_values > 1e-12]
    
    basis_values = np.diag(1/np.sqrt(basis_values))
    basis_vecs  = basis_vecs.dot(basis_values)
        
    L = np.zeros([n,n])
    for idx in range(len(classes)):
        L[labels==classes[idx], labels==classes[idx]] = 1/np.sum(labels==classes[idx])
    M = np.ones([m,m])/m
    H = (np.eye(m,m)-M).dot(basis_vecs).T.dot(K).dot(np.eye(m,m)-L)
    
    T = H.dot(H.T)
    
    eigenvecs = null(T)
    if eigenvecs.shape[1] < 1:
        eigenvals, eigenvecs = np.linalg.eig(T)
        eigenvals = np.diag(eigenvals)
        
    proj = (np.eye(m,m)-M).dot(basis_vecs).dot(eigenvecs)
    return proj
        
        
def center_kernel_matrix(kernel):
    '''
    Centers the data in the feature space only using the kernel matrix
    '''
    n = np.shape(kernel)[0]
    column_means = np.mean(kernel, 0)
    matrix_mean = np.mean(kernel)
    centered = kernel
    
    for idx in range(n):
        centered[idx, :] = centered[idx, :] - column_means
        centered[:, idx] = centered[:, idx] - column_means
        
    centered += matrix_mean
    return centered

