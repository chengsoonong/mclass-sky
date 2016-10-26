import numpy as np
import accpm
import scipy.optimize as opt

def linear_predictor(X, w):
    predictions = np.sign(np.dot(X, w))
    return predictions

def initial_polyhedron(X):
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

# def active(X, Y, M=None, iterations=None, maxiter=None):
#     (A, b) = initial_polyhedron(X)
#     i = 0
#     j = 0
#     while i < iterations:
#         print('\nEntering iteration', j) 
#         if j >= maxiter:
#             print('******** Maximum number of iterations reached ********')
#             return w_best
#         w_best = accpm.analytic_center(A, b)
#         (x_chosen, y_chosen) = query(A, b, X, Y, M)
#         print('    (x_chosen, y_chosen)  is', (x_chosen, y_chosen))
#         # if y_chosen * np.dot(w_best, x_chosen) < 0:
#         print('    y_chosen * np.dot(w_best, x_chosen) gives',
#               y_chosen * np.dot(w_best, x_chosen))
#         if y_chosen * np.dot(w_best, x_chosen) <= 0:
#             print('\n******** Cutting plane', i+1, 'added ********')
#             print('    w_best was', w_best)
#             a_cp = (-1 * y_chosen) * x_chosen
#             norm_a_cp = np.linalg.norm(a_cp)
#             a_cp = a_cp/norm_a_cp
#             b_cp = 0  
#             A = np.vstack((A, a_cp))
#             b = np.hstack((b, b_cp))
#             print('    w_best updated to', accpm.analytic_center(A, b))
#             i = i + 1
#         # if i == 1:
#         #     return (A, b)
#         j = j + 1
#     print('******** Desired number of points queried ********')    
#     return w_best

# def active(X, Y, iterations, center='ac', sample=1, M=None,
#            testing=0):

#     if center == 'ac':
#         center = accpm.analytic_center
#     if center == 'random':
#         center = random_vector

#     (A, b) = initial_polyhedron(X)
#     i = 0

#     if M == None:
#         M = 2*A.shape[1]

#     while i < iterations:

#         if testing == 2:
#             print('\nEntering iteration', i) 

#         w_best = center(A, b)
#         query_outcome = query(A, b, X, Y, M, 
#                               sample=sample, w_best=w_best)
#         (x_chosen, y_chosen) = query_outcome[0]
#         (X, Y) = query_outcome[1]

#         if testing == 2:
#             print('    (x_chosen, y_chosen)  is', (x_chosen, y_chosen))
#             print('    y_chosen * np.dot(w_best, x_chosen) gives',
#                   y_chosen * np.dot(w_best, x_chosen))
#             print('\n******** Cutting plane', i+1, 'added ********')
#             print('    w_best was', w_best)

#         a_cp = (-1 * y_chosen) * x_chosen
#         norm_a_cp = np.linalg.norm(a_cp)
#         a_cp = a_cp/norm_a_cp
#         b_cp = 0  
#         A = np.vstack((A, a_cp))
#         b = np.hstack((b, b_cp))

#         if testing == 2:
#             print('    w_best updated to', accpm.analytic_center(A, b))

#         i = i + 1
#     if testing == 1 or testing == 2:
#         print('******** Desired number of points queried ********')    
#     return w_best

def active(X, Y, iterations, center='ac', sample=1, testing=1, M=None):

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

def chebyshev_center(A, b):
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
    dimension = A.shape[1]
    not_feasible = True
    while not_feasible == True:
        rand_vec = np.random.uniform(-0.5, 0.5, dimension)
        if np.all(np.dot(A, rand_vec) <= b) == True:
            not_feasible = False
    return rand_vec

def query(A, b, X, Y, M, sample=1, w_best=None):
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
