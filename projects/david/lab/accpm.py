import numpy as np
import scipy.optimize as opt

def log_barrier(x, A, b):
    """
    Returns the value of the log barrier function associated with the 
    set of inequalities Ax <= b at x.
    
    It is preferable x, A, b are of type ndarray.
    """
    return -np.log(np.prod(b-np.dot(A, x)))

def logb_grad(x, A, b):
    """
    Returns the value of the gradient of the log barrier function 
    associated with the set of inequalities Ax <= b at x.
    """
    d = 1./(b-np.dot(A,x))
    return np.dot(np.transpose(A), d)

def logb_hess(x, A, b):
    """
    Returns the value of the Hessian of the log barrier function
    associated with the set of inequalities Ax <= b at x.
    """
    d = 1./(b-np.dot(A,x))
    diagd = np.diag(d)
    return np.dot(np.transpose(A), 
                  np.dot(np.linalg.matrix_power(diagd, 2), A))    

def phase_i_func(z, t, A, b):
    """
    Returns the value at z of the objective function of the phase I 
    optimization problem which has been approximated using the log barrier 
    function. 
    """
    x = z[:-1]
    x = np.reshape(x, (x.shape[0], 1))
    s = z[-1]
    # return t*s - np.log(np.prod(s + (b-np.dot(A, x))))
    return t*s - np.sum(np.log(s + (b-np.dot(A, x))))

def phase_i_grad(z, t, A, b):
    """
    Returns the gradient of the objective function at z of the phase I 
    optimization problem which has been approximated using the log barrier 
    function. 
    """
    x = z[:-1]
    s = z[-1]
    d = 1./(s + (b-np.dot(A, x)))
    grad_x = np.dot(np.transpose(A), d)
    grad_s = t - np.sum(d)
    grad = np.append(grad_x, grad_s)
    return grad

def phase_i_hess(z, t, A, b):
    """
    Returns the Hessian of the objective function at z of the phase I 
    optimization problem which has been approximated using the log barrier 
    function. 
    """
    x = z[:-1]
    s = z[-1] 
    d = 1./(s + (b-np.dot(A,x)))
    diagd = np.diag(d)
    d_sqd = np.power(d, 2)
    hess_xx = np.dot(np.dot(np.transpose(A), np.linalg.matrix_power(diagd, 2))
                     , A)
    hess_xs = -1 * np.dot(np.transpose(A), d_sqd)
    hess_sx = np.append(hess_xs, np.sum(d_sqd))
    hess_xs = np.reshape(hess_xs, (hess_xs.shape[0],1))
    hess = np.hstack((hess_xx, hess_xs))
    hess = np.vstack((hess, hess_sx))
    return hess

def phase_i_opt(A, b, t=4, mu=15, maxiter=2500):
    """
    Returns a point x that satisfies Ax<b, computed using the barrier method.
    """
    x = np.zeros((A[0].shape[0],1))
    s = 0.1*np.fabs(np.amax(-b)) + np.amax(-b)
    z = np.vstack((x, s))
    i = 0
    while i < maxiter and z[-1] >= 0:
        result = opt.minimize(phase_i_func, z, args = (t, A, b),
                              method='BFGS', jac=phase_i_grad)
        print(z)
        z = result.x
        t = mu*t
        i = i + 1
    print(i)
    return z

def analytic_center(A, b, z = None, epsilon=None, t=None, mu=None, maxiter=None):
    """
    Returns the analytic center of the inequalities Ax <= b.

    Parameters
    ----------------
    A : ndarray
    b : ndarray
        The matrix A and vector b for the inequalities Ax <= b.
    z : ndarray, optional
        The initial point for the optimization algorithm. Must satisfy
        Ax < b. If None, an initial point is computed.  
    """
    if z == None:
        z = phase_i_opt(A, b, t, mu, maxiter)
    result = opt.minimize(log_barrier, z, args=(A, b), method='Newton-CG', 
                          jac=logb_grad, hess = logb_hess)
    ac = result.x
    return ac        

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
    func_values = []
    feasible_values = []
    f_best = None
    while k < maxiter + 1:
        ac = analytic_center(polyhead)
        if ac == False: 
            return False
        func_k.append(func(ac))
        if stopping() == True:
            return (True, ac, func(ac), A, b, k)
        data = oracle(ac, func, grad, constr, epsilon)
        feasibility = data[0]
        func_values.append(func(ac))
        if feasibility == True:
            feasible_values.append(func(ac))
        cp = data[1]
        (a, b) = (cp[0], cp[1])
        (A, b) = (np.vstack(A, np.transpose(a)),
                  np.vstack(b, b))
        k = k + 1                                            