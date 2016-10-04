import numpy as np
import scipy.optimize as opt
import scipy.linalg as splinalg

# def log_barrier(x, A, b):
#     """
#     Returns the value of the log barrier function associated with the 
#     set of inequalities Ax <= b at x.
    
#     It is preferable x, A, b are of type ndarray.
#     """
#     return -np.log(np.prod(b-np.dot(A, x)))

# def logb_grad(x, A, b):
#     """
#     Returns the value of the gradient of the log barrier function 
#     associated with the set of inequalities Ax <= b at x.
#     """
#     d = 1./(b-np.dot(A,x))
#     return np.dot(np.transpose(A), d)

# def logb_hess(x, A, b):
#     """
#     Returns the value of the Hessian of the log barrier function
#     associated with the set of inequalities Ax <= b at x.
#     """
#     d = 1./(b-np.dot(A,x))
#     diagd = np.diag(d)
#     return np.dot(np.transpose(A), 
                  # np.dot(np.linalg.matrix_power(diagd, 2), A))    

# Analytic center computation: phase I problem and the Newton method.

# def phase_i_func(z, t, A, b):
#     """
#     Returns the value at z of the objective function of the phase I 
#     optimization problem which has been approximated using the log barrier 
#     function. 
#     """
#     x = z[:-1]
#     x = np.reshape(x, (x.shape[0], 1))
#     s = z[-1]
#     # return t*s - np.log(np.prod(s + (b-np.dot(A, x))))
#     return t*s - np.sum(np.log(s + (b-np.dot(A, x))))

# def phase_i_grad(z, t, A, b):
#     """
#     Returns the gradient of the objective function at z of the phase I 
#     optimization problem which has been approximated using the log barrier 
#     function. 
#     """
#     x = z[:-1]
#     s = z[-1]
#     d = 1./(s + (b-np.dot(A, x)))
#     grad_x = np.dot(np.transpose(A), d)
#     grad_s = t - np.sum(d)
#     grad = np.append(grad_x, grad_s)
#     return grad

# def phase_i_hess(z, t, A, b):
#     """
#     Returns the Hessian of the objective function at z of the phase I 
#     optimization problem which has been approximated using the log barrier 
#     function. 
#     """
#     x = z[:-1]
#     s = z[-1] 
#     d = 1./(s + (b-np.dot(A,x)))
#     diagd = np.diag(d)
#     d_sqd = np.power(d, 2)
#     hess_xx = np.dot(np.dot(np.transpose(A), np.linalg.matrix_power(diagd, 2))
#                      , A)
#     hess_xs = -1 * np.dot(np.transpose(A), d_sqd)
#     hess_sx = np.append(hess_xs, np.sum(d_sqd))
#     hess_xs = np.reshape(hess_xs, (hess_xs.shape[0],1))
#     hess = np.hstack((hess_xx, hess_xs))
#     hess = np.vstack((hess, hess_sx))
#     return hess

# def phase_i_opt(A, b, t=4, mu=15, maxiter=2500):
#     """
#     Returns a point x that satisfies Ax<b, computed using the barrier method.
#     """
#     x = np.zeros((A[0].shape[0],1))
#     s = 0.1*np.fabs(np.amax(-b)) + np.amax(-b)
#     z = np.vstack((x, s))
#     i = 0
#     while i < maxiter and z[-1] >= 0:
#         result = opt.minimize(phase_i_func, z, args = (t, A, b),
#                               method='BFGS', jac=phase_i_grad)
#         print(z)
#         z = result.x
#         t = mu*t
#         i = i + 1
#     print(i)
#     return z

# def analytic_center(A, b, z=None, epsilon=None, 
#                     t=None, mu=None, maxiter=None):
#     """
#     Returns the analytic center of the inequalities Ax <= b.

#     Parameters
#     ----------------
#     A : ndarray
#     b : ndarray
#         The matrix A and vector b for the inequalities Ax <= b.
#     z : ndarray, optional
#         The initial point for the optimization algorithm. Must satisfy
#         Ax < b. If None, an initial point is computed.  
#     epsilon : float, optional
#     t : float, optional
#     mu : float, optional
#     maxiter : int, optional
#     """
#     if z == None:
#         z = phase_i_opt(A, b, t, mu, maxiter)
#     result = opt.minimize(log_barrier, z, args=(A, b), method='Newton-CG', 
#                           jac=logb_grad, hess = logb_hess)
#     ac = result.x
#     return ac 

def initial(A, b):
    x = np.zeros(np.shape(A)[1])
    y = [0]*np.shape(A)[0]
    for i in range(len(y)):
        if b[i] > 0:
            y[i] = b[i]
        else: 
            y[i] = 1
    # print((x, np.asarray(y)))
    return (x, np.asarray(y))

def norm_res(x, y, v, A, b):
    g = -(1./y)
    r_d1 = np.dot(np.transpose(A), v)
    r_d2 = g + v
    r_p = y + np.dot(A, x) - b
    # print('r_d1=', r_d1)
    # print('r_d2=', r_d2)
    # print('r_p=', r_p)
    r = np.concatenate((r_d1, r_d2, r_p))
    return np.linalg.norm(r)

def newton_step(x, y, v, A, b):
    r_p = y + np.dot(A, x) - b
    g = -(1./y)
    H = np.diag(1./np.power(y, 2))
    A_chol = np.dot(np.dot(np.transpose(A), H),
                    A)
    b_chol = np.dot(np.transpose(A), g) - \
             np.dot(np.transpose(A), np.dot(H, r_p))
    L = np.linalg.cholesky(A_chol)
    # print('r_p =', r_p)
    # print('g =', g)
    # print('H =', H)
    # print('A_chol =', A_chol)
    # print('L =', L)
    # check = np.dot(L, np.transpose(L))
    # print('check=', check)
    z = splinalg.solve_triangular(L, b_chol, check_finite=False) 
    delta_x = splinalg.solve_triangular(np.transpose(L), z, check_finite=False)
    delta_y = np.dot(-A, delta_x) - r_p
    delta_v = np.dot(-H, delta_y) - g - v
    return (delta_x, delta_y, delta_v)

def analytic_center(A, b, x=None, y=None, rtol=10e-10, etol=10e-10, alpha=0.01, beta=0.99, 
                    maxiter=50):
    i = 0
    if x == None or y == None: 
        (x, y) = initial(A, b) 
    # print('x =', x)
    # print('y =', y)
    v = np.zeros(np.shape(A)[0])
    while i < maxiter:
        (delta_x, delta_y, delta_v) = newton_step(x, y, v, A, b)   
        t = 1.0
        zeros = np.zeros_like(y)
        while np.all(np.less_equal(y + t * delta_y, zeros)):
            t = beta*t
        while (norm_res(x + t*delta_x, y + t*delta_y, v + t*delta_v, A, b)) > \
              ((1 - alpha*t) * norm_res(x, y, v, A, b)):
            t = beta*t     
        (x, y, v) = (x + t*delta_x, y + t*delta_y, v + t*delta_v)
        if (np.linalg.norm(y + np.dot(A, x) - b) <= etol) and \
           (norm_res(x, y, v, A, b) <= rtol):
            # print('i =', i)
            return x
        i = i + 1  
    # print('i =', i)
    return x

def feasible(x, constr, epsilon):
    for i in range(len(constr)):
        fi_x = constr[i](x)
        if fi_x > 0:
            return (False, fi_x, 
                    opt.approx_fprime(x, constr[i], epsilon))
        else: 
            return (True, )
        
def oracle(x, func, grad, constr, epsilon):
    feasibility = feasible(x, constr, epsilon)[0]
    if feasibility == False:
        fi_x = feasibility[1]
        grad_fi_x = feasibility[2]
        (a, b) = (feasibility[2], 
                  np.dot(grad_fi_x, x) - fi_x)
    else:
        (a, b) = (grad(x), np.dot(grad(x), x))
    return (feasibility, (a, b))

def accpm(func, constr, epsilon, A, b, maxiter):
    """
    Solves the specified inequality constrained convex optimization 
    problem or feasbility problem via the analytic center cutting
    plane method (ACCPM). 
    
    Implementation applies to (inequality constrained) convex 
    optimization problems of the form
        minimize f_0(x)
        subject to f_i(x) <= 0, i = 1, ..., m,
    where f_0, ..., f_m are convex functions. The target set X is the
    epsilon-suboptimal set where epsilon is an argument of accpm.
    The ACCPM requires a set of linear inequality constraints,
    which represents a polyhedron, that all points in X satisfy. 
    That is, a matrix A and b that give constraints Ax <= b. 
    
    Parameters for convex optimization problems
    ----------------
    func : callable, func(x)
        The objective function f_0 to be minimized.
    constr : tuple with callable elements
        The required format is constr = (f_1, ..., f_m) where  
        f_i(x) is callable for i = 1, ...., m, 
        So constr[i-1](x) = f_i(x) for i = 1, ..., m.
    epsilon : float
        Specifies the value to be used for the target set,
        an epsilon-suboptimal set.
    A : ndarray
        Represents the matrix A.
    b : ndarray
        Represents the vector b. 
    maxiter : int, optional
        Maximum number of iterations to perform.
    """
    k = 0
    # func_values = [] 
    # feasible_values = []
    # f_best = None
    while k < maxiter + 1:
        ac = analytic_center(polyhead) # Compute the analytic center of
        if ac == False: 
            return False
        # func_k.append(func(ac))
        if stopping() == True:
            return (True, ac, func(ac), A, b, k)
        data = oracle(ac, func, grad, constr, epsilon)
        feasibility = data[0]
        func_values.append(func(ac))
        #if feasibility == True:
        #    feasible_values.append(func(ac))
        cp = data[1]
        (a, b) = (cp[0], cp[1])
        (A, b) = (np.vstack(A, np.transpose(a)),
                  np.vstack(b, b))
        k = k + 1                                            