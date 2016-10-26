import numpy as np
import accpm
import scipy.optimize as opt
import config

def linear_predictor(X, w):
    """
    Returns a vector of predictions (+1 or -1) given input data X and 
    parameter vector w.
    """
    predictions = np.sign(np.dot(X, w))
    return predictions

def initial_polyhedron(X):
    """
    Returns the initial polyhedron defined by Ax <= b, a unit hypercube
    centered at the origin.
    """
    dimension = X.shape[1]
    A = []
    b = []
    for i in range(dimension):
        a_upper = [0]*dimension
        a_lower = [0]*dimension
        a_upper[i] = 1
        a_lower[i] = -1
        A.append(a_upper)
        A.append(a_lower)
        b.append(0.5)
        b.append(0.5)
    A = np.array(A)
    b = np.array(b)
    return (A, b)

def chebyshev_center(A, b):
    """
    Computes the Chebyshev center of a polyhedron defined by Ax <= b.
    """
    dimension = A.shape[1] + 1

    bounds = []
    for i in range(dimension):
        bounds.append((None, None))

    c = np.zeros(dimension)
    c[-1] = -1
    norms = []
    for a_i in A:
        norm_a_i = np.linalg.norm(a_i)
        norms.append([norm_a_i])
    norms = np.asarray(norms)
    A = np.hstack((A, norms))
    result = opt.linprog(c, A, b, bounds=bounds)
    cc = result.x[:-1]
    return cc

def random_vector(A, b):
    """
    Generates a random vector satisfying Ax <= b through rejection
    sampling.
    """

    dimension = A.shape[1]
    not_feasible = True
    while not_feasible == True:

        config.reject_counter = config.reject_counter + 1
        if config.reject_counter == config.milestone:
            config.milestone = config.milestone * 10
            print(config.reject_counter, 'random vectors have been generated so far')

        rand_vec = np.random.uniform(-0.5, 0.5, dimension)
        if np.all(np.dot(A, rand_vec) <= b) == True:
            not_feasible = False
    return rand_vec

def query(A, b, X, Y, M, sample=1, w_best=None):
    """
    Chooses a training pattern to have its label examined.

    Parameters
    ----------------
    A : ndarray
    b : ndarray
        Specifies the polyhedron defined by Ax <= b.
    X : ndarray
        Training data that the pattern to have its label examined is
        chosen from.
    Y : ndarray
        Labels of the training data to be used for labelling.
    M : int, optional
        Specifies the number of points to sample from the polyhedron. 
        By default this is taken to be the number of features squared, 
        which is passed from the active function.
    sample : 0, 1, optional
        Specifies how the center of the polyhedron will be 
        approximated.
            0 - w_best is used for this purpose. 
            1 (default) - M points are uniformly sampled from the 
                          polyhedron and averaged.
    w_best : ndarray, optional
        If sample = 1, then w_best must be specified.

    Returns
    ----------------
    (x_chosen, y_chosen) : tuple
        The training pattern chosen and its label.
    (X, Y) : tuple
        The data set X and Y with x_chosen and y_chosen removed, 
        respectively.
    """
    if sample == 1:
        dimension = X.shape[1]
        sum = np.zeros(dimension)
        for i in range(M):
            rand_vec = random_vector(A, b)
            sum = sum + rand_vec
        g = sum/M

    if sample == 0:
        g = w_best

    min_val = np.inf
    ind = 0
    for i in range(X.shape[0]):
        current_val = np.dot(g, X[i])
        if current_val < min_val:
            ind = i
            min_val = current_val
    x_chosen = X[ind]
    y_chosen = Y[ind]
    X  = np.delete(X, ind, axis=0)
    Y = np.delete(Y, ind, axis=0)

    return ((x_chosen, y_chosen), (X, Y)) 

def active(X, Y, iterations, center='ac', sample=1, testing=1, M=None):
    """
    Computes the parameter vector for linear_predictor using a cutting
    plane active learning procedure.

    Parameters
    ----------------
    X : ndarray
        Training data. If iterations = n, then the active learning
        procedure will choose n training patterns to be labelled.
    Y : ndarray
        Labels of training data. If iterations = n, then only n labels
        will be used.
    iterations : int
        The number of points chosen to have their label examined. Must
        be less than or equal to the number of training patterns.
    center : 'ac', 'cc', 'random', optional
        Specifies how, at each iteration, the center of the polyhedron 
        is to be computed. 
            'ac' (default) - analytic center
            'cc' - Chebyshev center
            'random' - random center 
    sample : 0, 1 (default), optional
        Specifies how the center of the polyhedron will be 
        approximated in the query function.
    testing : 0, 1 (default), 2, 3, optional
        Specifies the information to be returned and to be printed as
        the procedure runs.
            0 - returns w_best only.
            1 - returns w_best only and prints success summary.
            2 - returns w_best only, prints success summary and 
                prints information at each iteration.
            3 - returns w_best, the number j of cutting planes
                generated and the array of iterations = n parameter
                vectors generated.
    M : int, optional
        Specifies the number of points to sample from the polyhedron in
        the query function. By default this is taken to be the number 
        of features squared.

    Returns
    ----------------
    w_best : ndarray
        The parameter computed on the final iteration.
    (Only when testing = 3) j : int
        The number of cutting planes generated.
    (Only when testing = 3) weights : list
        List containing the iterations = n parameters computed.

    """


    if center == 'ac':
        center = accpm.analytic_center
    if center == 'cc':
        center = chebyshev_center
    if center == 'random':
        center = random_vector

    (A, b) = initial_polyhedron(X)
    weights = []
    i = 0
    j = 0

    if M == None:
        M = A.shape[1]*A.shape[1]

    while i < iterations:

        if testing == 2:
            print('\nEntering iteration', i) 

        w_best = center(A, b)
        weights.append(w_best)
        query_outcome = query(A, b, X, Y, M, 
                              sample=sample, w_best=w_best)
        (x_chosen, y_chosen) = query_outcome[0]
        (X, Y) = query_outcome[1]

        if testing == 2:
            print('    (x_chosen, y_chosen)  is', (x_chosen, y_chosen))
            print('    y_chosen * np.dot(w_best, x_chosen) gives',
                  y_chosen * np.dot(w_best, x_chosen))
        
        if y_chosen * np.dot(w_best, x_chosen) <= 0:
            a_cp = (-1 * y_chosen) * x_chosen
            norm_a_cp = np.linalg.norm(a_cp)
            a_cp = a_cp/norm_a_cp
            b_cp = 0  
            A = np.vstack((A, a_cp))
            b = np.hstack((b, b_cp))

            j = j + 1

            if testing == 2:
                print('\n******** Cutting plane', i+1, 'added ********')
                print('    w_best was', w_best)
                print('    w_best updated to', accpm.analytic_center(A, b))
                print('    a_cp is', a_cp)

        i = i + 1

    if testing == 3:
        return (w_best, j, weights)

    if testing == 1 or testing == 2:
        print('******** Desired number of points queried ********')
        print('   ', j, 'cutting plane(s) generated over', i, 'iterations') 
        return w_best

    if testing == 0:
        return w_best