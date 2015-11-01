import numpy as np

def squared_euclidean_distances(x, y):
    n = np.shape(x)[0]
    m = np.shape(y)[0]
    distmat = np.zeros((n,m))
    
    for i in range(n):
        for j in range(m):
            buff = x[i,:] - y[j,:]
            distmat[i,j] = buff.dot(buff.T)
    return distmat

def score(proj, target_points, ks):
    projection_vectors = ks.T.dot(proj)
    sq_dist = squared_euclidean_distances(projection_vectors, target_points)
    scores = np.sqrt(np.amin(sq_dist, 1))
    return scores