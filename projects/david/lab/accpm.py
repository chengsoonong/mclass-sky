import numpy as np
import scipy.optimize as opt
import scipy.linalg as splinalg

myfloat = np.float64
np.set_printoptions(precision=2)

def logb_grad(x, A, b):
    """
    Returns the value of the gradient of the log barrier function 
    associated with the set of inequalities Ax <= b at x.
    """
    d = 1./(b-np.dot(A,x))
    return np.dot(np.transpose(A), d) 

def is_positive(x):
    """
    Returns True if every entry of the vector x is positive. 
    """
    zeros = np.zeros_like(x)
    return np.all(np.greater(x, zeros))

def start0(A, b, x, y):
    """
    Generates starting points for the infeasible start Newton method
    when start = 0.
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
    Generates starting points for the infeasible start Newton method
    when start = 1, which is by default.
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
                # if b[i] - np.dot(A[i], x0) > 0:
                if b[i] - np.dot(A[i], x0) > 1e10-15:
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
    r = np.concatenate((r_d1, r_d2, r_p))
    return np.linalg.norm(r)

def newton_step(x, y, v, A, b, testing=0):
    """
    Computes the Newton step (delta_x, delta_y, delta_v). 
    """ 
    r_p = y + np.dot(A, x) - b
    g = -(1./y)
    H = np.diag(1./np.power(y, 2))
    A_chol = np.dot(np.dot(np.transpose(A), H), A)
    
    if testing == 2:
        print('Computing the Newton step:')
        print('    H has diag', 1./np.power(y, 2))
        print('    A_chol has eigenvalues:', splinalg.eigvalsh(A_chol))
        print('    A_chol is\n', A_chol)

    b_chol = np.dot(np.transpose(A), g) - \
             np.dot(np.transpose(A), np.dot(H, r_p))
    L = splinalg.cholesky(A_chol, lower=True)
    if np.linalg.norm(A_chol - np.dot(L, np.transpose(L))) > 10e-6:
        print('**** Cholesky factorization FAILED or INACCURATE ****')
    z = splinalg.solve_triangular(L, b_chol, check_finite=False) 
    delta_x = splinalg.solve_triangular(np.transpose(L), z, check_finite=False)
    delta_y = np.dot(-A, delta_x) - r_p
    delta_v = np.dot(-H, delta_y) - g - v
    return (delta_x, delta_y, delta_v)

def analytic_center(A, b, x0=None, y0=None, rtol=10e-6, etol=10e-6, 
                    start=0, alpha=0.01, beta=0.5,
                    testing=0, maxiter=50):
    """
    Computes the analytic center on the inequalities Ax <= b.
    """

    i = 0
    if start == 0:
        (x0, y0) = start0(A, b, x0, y0)
    if start == 1:
        (x0, y0) = start1(A, b, x0, y0)
    (x, y) = (x0, y0)

    if testing == 2:
        print('Computing the analytic center:')
        print('    A is\n', A)
        print('    b is', b)
        print('    x0 =', x)
        print('    y0 =', y)
    
    v = np.zeros(np.shape(A)[0])
    while i < maxiter:
        (delta_x, delta_y, delta_v) = newton_step(x, y, v, A, b, 
                                                  testing=testing)  
        # print('at iteration', i,'delta_x is\n', delta_x) 
        # print('    delta_x=', delta_x)
        # print('    delta_y=', delta_y)
        # print('    delta_v=', delta_v)
        t = 1.0
        zeros = np.zeros_like(y, dtype=myfloat)
        # while np.all(np.less_equal(y + t * delta_y, zeros)):
        while np.any(np.less_equal(y + t * delta_y, zeros)):
            t = beta*t
        while (norm_res(x + t*delta_x, y + t*delta_y, v + t*delta_v, A, b)) > \
              ((1 - alpha*t) * norm_res(x, y, v, A, b)):
            t = beta*t     
        # while (np.all(np.less_equal(y + t * delta_y, zeros))) and \ 
        #       ((norm_res(x + t*delta_x, y + t*delta_y, v + t*delta_v, A, b)) > \
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
    if constr==None:
        return True
    for i in range(len(constr)):
        fi_x = constr[i](x)
        if fi_x > 0:
            return i 
    return True
        
def oracle(ac, func, grad_func, fbest, args=(), constr=None, grad_constr=None):
    """
    Returns ((a, b), fbest) where (a, b) specifies the cutting plane 
    a.x <= b and fbest is the lowest objective value encountered so far.
    """
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
        func_ac = func(ac, *args)
        grad_func_ac = grad_func(ac, *args)
        if func_ac <= fbest:
            fbest = func_ac 
        a = grad_func_ac
        b = np.dot(grad_func_ac, ac) - func_ac + fbest
        return ((a, b), fbest)

def normalize(A, b):
    """
    Normalizes the inequalities Ax <= b. That is, divides each  
    inequality ai.x <= bi by norm(ai).
    """
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
    A_normalized = np.asarray(A_normalized, dtype=myfloat)
    b_normalized = np.asarray(b_normalized, dtype=myfloat)
    return (A_normalized, b_normalized)

def accpm(A, b, func, grad_func, constr=None, grad_constr=None, args=(), 
          alpha=0.01, beta=0.7, x0=None, y0=None, start=1, 
          tol=10e-3, maxiter=50,
          summary=1, testing=0):
    """
    Solves the specified inequality constrained convex optimization 
    problem or feasbility problem via the analytic center cutting
    plane method (ACCPM). 
    
    Implementation applies to (inequality constrained or unconstrained)
    convex optimization problems of the form
        minimize f_obj(x)
        subject to f_i(x) <= 0, i = 0, ..., m,
    where f_obj, f_0, ..., f_m are convex differentiable functions. 
    The ACCPM requires a set of initial linear inequality constraints 
    that represent the initial polyhedron in which to search for 
    satisfcatory solutions. That is, a matrix A and b that give 
    constraints Ax <= b. The algorithm terminates when a point x
    is found that satisfies norm(grad f_obj(x)) <= tol.

    
    Parameters
    ----------------
    A : ndarray
        Represents the matrix A.
    b : ndarray
        Represents the vector b.
    func : callable, func(x, *args)
        The objective function f_obj to be minimized.
    grad_func : callable, grad_func(x, *args)
        The gradient of the objective function f_obj.
    constr : tuple (or list) with callable elements, optional
        The required format is constr = (f_0, ..., f_m) where  
        f_i(x) is callable for i = 0, ..., m. 
        So constr[i](x) = f_i(x) for i = 1, ..., m.
        Similarly if a list.
    grad_constr : tuple (or list) with callable elements, optional
        The required format is constr = (gradf_0, ..., gradf_m) where  
        gradf_i(x) is callable for i = 0, ..., m. 
    args : tuple, optional
        Optional arguments for func and grad_func.
    alpha : float, optional
    beta : float, optional
        Parameters for the ACCPM. Default values are alpha = 0.01 and
        beta = 0.7, which are sufficient for most applications. 
    x0 : ndarray, optional 
    y0 : ndarray, optional
        The initial points x0 and y0 to be used for the generation of
        the AC on the 0th iteration. If x0 = y = None, points will be
        generated.
    start : 0, 1, optional
        Specifies how starting points for the AC omputation, via the 
        infeasible start Newton method, will be chosen.  Default is 
        0 where x0 = 0 and y0 is chosen accordingly. If 1, then 
        x0 = x_prev where x_prev is the AC generated during 
        the previous iteration and y0 is chosen accordingly. 
    tol : float, optional
        Specifies the tolerance of the algorithm. 
    maxiter : int, optional
        Maximum number of iterations to perform.
    summary : int, 0, 1, optional
        If 0 will not print summary of results. If 2 will print summary
        of results.
    testing : int, 0, 1, 2, optional
        If 0 will not print testing results. If 1 brief testing results
        will be printed. If 2 detailed testing results will be printed.

    Returns
    ----------------
    outcome : bool
        True if ACCPM succeeded. False otherwise.
    ac : ndarray
        The last AC calculated. This is the solution if outcome = True.
    value_attained : float
        The value of f_obj at the last ac calculated. 
    iterations : int
        The number of iterations performed which is the number of
        analytic centers generated.
    (Only when outcome = False) fbest : float
        The minimum value attained by f_obj over all iterations.
    """

    # TO-DO:
    # - Incorporate checking of lower bound.
    # - Description of what is returned.
    # - Return useful information. 

    (A, b) = normalize(A, b)
    k = 0
    fbest = np.inf

    if testing != 0:
        print('-------- Starting ACCPM --------')
        print('Initially: b =', b, 'and A =\n', A)
        print('--------------------------------')

    while k < maxiter:

        if testing != 0:
            print('Entering iteration', k)

        ac = analytic_center(A, b, x0=x0, y0=y0, 
                             alpha=alpha, beta=beta, start=start, 
                             testing=testing)
        (x0, y0) = (ac, None)

        if np.linalg.norm(grad_func(ac, *args)) <= tol:
            value_attained = func(ac, *args)
            outcome = True
            iterations = k + 1
            if summary == 1:
                print('******** ACCPM SUCCEEDED ********')
                print('    Solution point:', ac) 
                print('    Objective value:', value_attained)
                print('    Iterations:', iterations)
            return (outcome, ac, value_attained, iterations)

        data = oracle(ac, func, grad_func, fbest, args=args, 
                      constr=constr, grad_constr=grad_constr)
        (a_cp, b_cp) = data[0]
        norm_a_cp = np.linalg.norm(a_cp)
        a_cp = a_cp/norm_a_cp
        b_cp = b_cp/norm_a_cp
        fbest = data[1]

        if testing == 1:
            if (np.linalg.norm(logb_grad(ac, A, b)) <= tol):
                print('At iteration', k, 'AC computation SUCCEEDED with AC', ac, 'where')
                print('        a_cp =', a_cp, 'and', 'b_cp =', np.array([b_cp]))
                print('--------------------------------')
            else:
                print('At iteration', k, 'AC computation FAILED with AC', ac, 'where')
                print('        a_cp =', a_cp, 'and', 'b_cp =', np.array([b_cp]))
                print('--------------------------------')

        A = np.vstack((A, a_cp))
        b = np.hstack((b, b_cp))
        k = k + 1

    value_attained = func(ac, *args)
    outcome = False
    iterations = k + 1
    if summary == 1:
        print('******** ACCPM FAILED ********')
        print('    Last AC:', ac)
        print('    Objective value at last AC:', value_attained)
        print('    Iterations:', iterations)
        print('    Best objective value was:', fbest)
        
    return (outcome, ac, value_attained, k, fbest)