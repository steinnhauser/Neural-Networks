import numpy as np
import sklearn

def logistic_regression():
    # need to determine:
    #   Cost function
    #   Design matrix

    # Implement a gradient descent solver. Either:
    #   Standartd GD with learning rate, or
    #   Attempt Newton-Raphson method.

    # May be useful to implement a Stochastic Gradient Descent method which has
    # an argument "mini-batches". This is useful for the Neural Network later.
    pass

def gradient_descent_solver(X, y, x0=0, random_state_x0=False,\
    gamma_k = 0.1, max_iter=50, tol=1e-2):
    """
    Calculates a gradient descent starting from x0.

    Parameters:
    -----------
    x0 : vec
        Initial guess for the minimum of F
    gamma_k : float
        Learning rate of the solver ('step size' of delF).
    max_iter : int
        Maximum amount of iterations before exiting.
    tol : float
        Tolerance which dictates when an answer is sufficient.

    Returns:
    --------
    xsol : vec
        Vector which produces a minimum of F.
    """
    if gamma_k <= 0:
        raise ValueError("Bad useage:\n\tThe learning rate is negative.")

    if random_state_x0:
        preds = X.shape[1] # p
        xsol = (np.random.random(preds) - 0.5)*0.7/0.5 # between [-0.7, 0.7]
        if not type(x0) == int:
            if not np.equal(x0.all(), 0):
                print("Useage Warning: Overwriting set x0 with random values.")
        elif type(x0) == int and x0!=0:
            print(f"Useage Warning: Overwriting set x0={x0} with random values")

    else:
        if type(x0) == int:
            if x0:
                xsol = np.ones(X.shape[1])*x0
            else:
                xsol=np.zeros(X.shape[1])
        elif type(x0) == np.ndarray:
            if x0.shape[0]==X.shape[1]: # if len = p
                xsol = x0
            else:
                raise ValueError("Bad useage: x0 was not of length 1 or p.")
        else:
            raise\
        ValueError("Bad useage: x0 was not of type 'int' or 'numpy.ndarray'")

    # calculate the first step
    p = calculate_p(X, xsol)
    dF = delF(X, y, p)
    step = gamma_k * dF

    i = 0
    while i <= max_iter:
        xsol = xsol - step
        # calculate the next step
        p = calculate_p(X, xsol)
        dF = delF(X, y, p)
        step = gamma_k*dF
        if np.linalg.norm(step) <= tol:
            print("GD reached tolerance.")
            break
        i += 1

    if i >= max_iter:
        print("GD reached max iteration.")

    return xsol

def delF(X, y, p):
    """
    Calculates an estimation of the gradient of the cost function F.

    Parameters:
    -----------
    x : vec
        Input which dictates where gradient should work from

    Returns:
    --------
    dF : vec
        Output of which direction F decreases in.
    """
    a = y - p
    dF = - X.T @ a
    return dF

def calculate_p(X, xsol):
    """
    Calculates the probability vector using the sigmoid function.
    """
    fac = X @ xsol
    p = 1./(1+np.exp(-fac)) # np.exp(fac)(1+np.exp(fac))is strange
    return p

def CGMethod(X, y, x0=0, random_state_x0=False,\
    gamma_k = 0.01, max_iter=50, tol=1e-3):
    """
    Conjugate Gradient method for finding solution of non-linear problems.
    Reduces residual error r=b-Ax iteratively. Exact solution yields r=0.
    Constrain that the matrix A is positive definite and symmetric.

    Parameters:
    -----------
    X : mtx
        (N x p) matrix of predictors
    y : vec
        (N x 1) vector of targets
    x0 : vec, default np.zeros(p)
        (p x 1) Initial guess solution.
    random_state_x0 : bool, default False
        If True, set x0 to have elements in [-0.7, 0.7] randomly.
    gamma_k : float, default 0.01
        Learning rate for the descent.
    max_iter : int, default 50
        Maximum amount of iterations which the gradient descent will perform.
    tol : float, default 0.001
        Tolerance for which the iterative process will continue.

    Returns:
    --------
    xsol : vec
        (p x 1) vector solution to b - Ax
    """

    if random_state_x0:
        len = X.shape[1] # p
        xsol = (np.random.random(len) - 0.5)*0.7/0.5 # between [-0.7, 0.7]
        if x0 != 0:
            print(f"Warning:\n\tRandom state is overwriting the set x0={x0}.")
    else:
        xsol = x0

    i=0
    while i<max_iter:
        # A = What is its relation to X?
        r = y - A @ xsol
        den = r.T@r
        num = r.T @ (A @ r)
        alpha = (den)/(num)
        xsol -= alpha @ r
        if np.linalg.norm(xsol) < tol:
            print(f"Tolerance {tol} reached.")
            break
        i+=1

    if i>=max_iter:
        print(f"Max iteration {i} reached.")

    return xsol

if __name__ == '__main__':
    pass
    # logistic_regression()
    # CGMethods()
