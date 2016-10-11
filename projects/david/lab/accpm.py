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

def logb_grad(x, A, b):
    """
    Returns the value of the gradient of the log barrier function 
    associated with the set of inequalities Ax <= b at x.
    """
    d = 1./(b-np.dot(A,x))
    return np.dot(np.transpose(A), d)

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
def is_positive(x):
    """
    Returns True if every entry of the vector x is positive. 
    """
    zeros = np.zeros_like(x)
    return np.all(np.greater(x, zeros))

def start0(A, b, x, y):
    """
    Generates starting points for the infeasible start Newton method.
    """
    x = np.zeros(np.shape(A)[1])
    y = [0]*np.shape(A)[0]
    for i in range(len(y)):
        if b[i] > 0:
            y[i] = b[i]
        else: 
            y[i] = 1
    # print((x, np.asarray(y)))
    return (x, np.asarray(y))

def start1(A, b, x0, y0):
    """
    Generates starting points for the infeasible start Newton method.
    """
    if y0 is not None:
        if is_positive(y0) == False:
            print('y0 is not positive!')
            return (None, None)

    if x0 is None: 
        x0 = np.zeros(np.shape(A)[1])
        if y0 is None: 
            y0 = [0]*np.shape(A)[0]
            for i in range(len(y0)):
                if b[i] > 0:
                    y0[i] = b[i]
                else: 
                    y0[i] = 1
            y0 = np.asarray(y0)
            return (x0, y0)
        else: 
            return (x0, y0)      
    else: 
        if y0 is None:
            y0 = [0]*np.shape(A)[0]
            for i in range(len(y0)):
                if b[i] - np.dot(A[i], x0) > 0:
                    y0[i] = b[i] - np.dot(A[i], x0)
                else: 
                    y0[i] = 1
            y0 = np.asarray(y0)
            return (x0, y0)
        else: 
            return (x0, y0)

def norm_res(x, y, v, A, b):
    """
    Computes the norm of the vector of residuals 
    (r_dual(x, y, v), r_primal(x, y, v)).
    """
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
    """
    Computes the Newton step (delta_x, delta_y, delta_v). 
    """
    r_p = y + np.dot(A, x) - b
    g = -(1./y)
    H = np.diag(1./np.power(y, 2))
    A_chol = np.dot(np.dot(np.transpose(A), H), A)
    b_chol = np.dot(np.transpose(A), g) - \
             np.dot(np.transpose(A), np.dot(H, r_p))
    L = np.linalg.cholesky(A_chol)
    # print('Shape of A_chol is', A_chol.shape)
    # print('Shape of L is', L.shape)
    # print('Shape of L^T is', np.transpose(L).shape)
    if np.linalg.norm(A_chol - np.dot(L, np.transpose(L))) > 10e-6:
        print('>>>> Cholesky factorization failed or inaccurate! <<<<')
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

def analytic_center(A, b, x0=None, y0=None, rtol=10e-5, etol=10e-5, 
                    start=0, alpha=0.01, beta=0.5,
                    maxiter=50):
    """
    Computes the analytic center on the inequalities Ax <= b.
    """
    i = 0
    if start == 0:
        (x0, y0) = start0(A, b, x0, y0)
    if start == 1:
        (x0, y0) = start1(A, b, x0, y0)
    (x, y) = (x0, y0)
    # print('    x =', x)
    # print('    y =', y)
    v = np.zeros(np.shape(A)[0])
    while i < maxiter:
        (delta_x, delta_y, delta_v) = newton_step(x, y, v, A, b)  
        # print('at iteration', i,'delta_x is\n', delta_x) 
        # print('    delta_x=', delta_x)
        # print('    delta_y=', delta_y)
        # print('    delta_v=', delta_v)
        t = 1.0
        zeros = np.zeros_like(y)
        # while np.all(np.less_equal(y + t * delta_y, zeros)):
        while np.any(np.less_equal(y + t * delta_y, zeros)):
            t = beta*t
        while (norm_res(x + t*delta_x, y + t*delta_y, v + t*delta_v, A, b)) > \
              ((1 - alpha*t) * norm_res(x, y, v, A, b)):
            t = beta*t     
        # while (np.all(np.less_equal(y + t * delta_y, zeros))) and ((norm_res(x + t*delta_x, y + t*delta_y, v + t*delta_v, A, b)) > \
        #       ((1 - alpha*t) * norm_res(x, y, v, A, b))):
        #     t = beta*t    
        (x, y, v) = (x + t*delta_x, y + t*delta_y, v + t*delta_v)
        # print('    At iteration', i, 'x is', x, 'and y is', y, 'with t=', t)
        if (np.linalg.norm(y + np.dot(A, x) - b) <= etol) and \
           (norm_res(x, y, v, A, b) <= rtol):
            #print('    SUCCESS with i =', i)
            return x
        i = i + 1  
    #print('    FAILURE with i =', i)
    return x

def feasible(x, constr):
    """
    Checks the inequality constraints at x. If x is a feasible point,
    returns True. If it is infeasible, returns the index of the first
    constraint violated.
    """

    # TO-DO:
    # - Re-write more elagently.
    for i in range(len(constr)):
        fi_x = constr[i](x)
        if fi_x > 0:
            return i 
    return True
        
def oracle(ac, func, constr, grad_func, grad_constr, fbest):
    """
    Returns ((a, b), fbest) where (a, b) specifies the cutting plane 
    a.x <= b and fbest is the lowest objective value encountered so far.
    """

    # TO-DO: 
    # - What if constr[i] == None? 
    feasibility = feasible(ac, constr)
    if feasibility != True:
        i = feasibility
        fi = constr[i]
        grad_fi = grad_constr[i]
        fi_ac = fi(ac)
        grad_fi_ac = grad_fi(ac) 
        a = grad_fi_ac
        b = np.dot(grad_fi_ac, ac) - fi_ac
        return ((a, b), fbest)
    else:
        func_ac = func(ac)
        grad_func_ac = grad_func(ac)
        if func_ac <= fbest:
            fbest = func_ac 
        a = grad_func_ac
        b = np.dot(grad_func_ac, ac) - func_ac + fbest
        return ((a, b), fbest)

def normalize(A, b):
    A_normalized = []
    b_normalized  = []
    for i in range(np.shape(A)[0]):
        Ai = A[i]
        bi = b[i]
        norm_Ai = np.linalg.norm(Ai)
        normalized_Ai = Ai/norm_Ai
        normalized_bi = bi/norm_Ai
        A_normalized.append(normalized_Ai)
        b_normalized.append(normalized_bi)
    A_normalized = np.asarray(A_normalized)
    b_normalized = np.asarray(b_normalized)
    return (A_normalized, b_normalized)

def accpm(func, constr, A, b, x0=None, y0=None, 
          grad_func=None, grad_constr=None, 
          alpha=0.01, beta=0.7, start=0, tol=10e-4, maxiter=50,
          testing=False):
    """
    Solves the specified inequality constrained convex optimization 
    problem or feasbility problem via the analytic center cutting
    plane method (ACCPM). 
    
    Implementation applies to (inequality constrained) convex 
    optimization problems of the form
        minimize f_obj(x)
        subject to f_i(x) <= 0, i = 0, ..., m,
    where f_obj, f_0, ..., f_m are convex differentiable functions. 
    The target set X is the epsilon-suboptimal set where epsilon is 
    exactly specified by the argument passed to the tol parameter.
    The ACCPM requires a set of linear inequality constraints, 
    which represents a polyhedron, that all points in X satisfy. 
    That is, a matrix A and b that give constraints Ax <= b. 
    
    Parameters for convex optimization problems
    ----------------
    func : callable, func(x)
        The objective function f_obj to be minimized.
    constr : tuple (or list) with callable elements
        The required format is constr = (f_0, ..., f_m) where  
        f_i(x) is callable for i = 0, ..., m. 
        So constr[i](x) = f_i(x) for i = 1, ..., m.
        Similarly if a list.
    A : ndarray
        Represents the matrix A.
    b : ndarray
        Represents the vector b. 
    x0 : ndarray, optional 
    y0 : ndarray, optional
        The initial points x0 and y0 to be used for the generation of
        the AC on the 0th iteration. 
    grad_func : callable, grad_func(x)
        The gradient of the objective function f_obj.
    grad_constr : tuple (or list) with callable elements
        The required format is constr = (gradf_0, ..., gradf_m) where  
        gradf_i(x) is callable for i = 0, ..., m.
    start : 0 or 1, optional
        Specifies how initial (or starting) points for the AC 
        computation, via the infeasible start Newton method, will be 
        chosen.  Default is 0, x0 = 0 and y0 is chosen accordingly. If
        1, then x0 = x_prev where x_prev is the AC generated during 
        the previous iteration. 
    tol : float, optional
        Specifies the value of epsilon to be used for the 
        epsilon-suboptimal target set.
    maxiter : int, optional
        Maximum number of iterations to perform.
    """

    # TO-DO:
    # - Extend parameters to account for optional arguments also.
    # - Incorporate checking of lower bound.
    # - Description of what is returned.
    # - Return useful information. 

    (A, b) = normalize(A, b)
    k = 0
    fbest = np.inf

    if testing == True:
        np.set_printoptions(precision=4)
        print('Initially: b =', b, 'and A =\n', A)
        print('----------------')

    while k < maxiter:
        # print('At iteration', k, )
        ac = analytic_center(A, b, x0=x0, y0=y0, alpha=alpha, beta=beta, start=start)
        (x0, y0) = (ac, None)
        # print('    AC is', ac)

        all_zeros = not np.any(grad_func(ac))
        if all_zeros == True:
            print('Success!')
            return ac
        data = oracle(ac, func, constr, grad_func, grad_constr, fbest)
        (a_cp, b_cp) = data[0]
        norm_a_cp = np.linalg.norm(a_cp)
        a_cp = a_cp/norm_a_cp
        b_cp = b_cp/norm_a_cp
        fbest = data[1]

        if testing == True:
            if (np.linalg.norm(logb_grad(ac, A, b)) <= 10e-4):
                print('At iteration', k, 'AC computation SUCCEEDED with AC', ac, 'where')
                print('        a_cp =', a_cp, 'and', 'b_cp =', np.array([b_cp]))
            else:
                print('At iteration', k, 'AC computation FAILED with AC', ac, 'where')
                print('        a_cp =', a_cp, 'and', 'b_cp =', np.array([b_cp]))

        A = np.vstack((A, a_cp))
        b = np.hstack((b, b_cp))
        k = k + 1 
        # print('A is\n', A)
        # print('b is', b)
    print('******** ACCPM failed! ********') 