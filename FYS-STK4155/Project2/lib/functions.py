import pandas as pd
import scipy
import sklearn
import numpy as np
from sklearn.linear_model import SGDRegressor


def read_in_data(fn, headers=False, shuffle=False, seed=0, scale=True):
    """
    Reads in xls files using pandas and sorts dataframes.

    Parameters:
    -----------
    fn : str
        Data filename (+ pathway from the Project2 directory).
    headers : bool, default False
        Returns headers (row #1) as a list of strings.
    shuffle : bool, default False
        Shuffles the datasets X and y by their rows.
    seed : int
        Integer for a random number generator if shuffle=True
    scale : bool, default False
        Scales the X and y values to be stochastic.
        If True, then the intercept=False is required for Regression.

    Returns:
    --------
    X : mtx
        (N x p) matrix of predictors
    y : vec
        (N x 1) vector of outcomes
    headers : list
        (1 x p) list of headers
    """

    if __name__ == '__main__':
        filename = "../" + fn
    else:
        filename = fn

    df = pd.read_excel(filename) # shape (30001, 25)
    # first column is indices, last column is whether they default or not (0/1)
    # first row is headers classifying the X matrix.
    X = np.array((df.values[1:,1:]), dtype=int)
    y = np.array(df.values[1:,-1], dtype=int)

    if shuffle:
        X, y = shuffle_Xy(X, y, seed)

    if scale:
        X = sklearn.preprocessing.scale(X)
        y = sklearn.preprocessing.scale(y)

    if headers:
        headers = df.values[0,1:-2] # headers of X-columns in the same order.
        return X, y, headers
    else:
        return X, y

def shuffle_Xy(X, y, seed):
    """
    Shuffles a (N x p) matrix X and (N x 1) vector y row-wise.

    Parameters:
    -----------
    X : mtx
        (N x p) matrix
    y : vec
        (N x 1) vector
    seed : int
        int for the random number generator

    Returns:
    --------
    X : mtx
        Shuffled matrix X row-wise
    y : vec
        Shuffled vector y row-wise
    """

    X_sparse = scipy.sparse.coo_matrix(X) # a sparse matrix in coordinate format
    X, X_sparse, y = sklearn.utils.shuffle(X, X_sparse, y, random_state=seed)
    return X, y

def sklearn_GDRegressor(X, y, intercept=False):
    """
    Uses scikit-learn's Gradient descent method
    to calculate a minimum of a cost function (MSE by default).

    Parameters:
    -----------
    X : mtx
        (N x p) matrix
    y : vec
        (N x 1) vector
    intercept : bool, default False
        Decides whether the SGDRegressor fits the intercept.

    Returns:
    --------
    clf : SGDRegressor object
        Classifier for data X and y, where clf.coef_ = 'beta'
    """
    clf = sklearn.linear_model.SGDRegressor(penalty='none',learning_rate=\
        'constant', eta0=0.01, max_iter=50, fit_intercept=intercept)
    clf.fit(X,y)
    return clf

def assert_binary_accuracy(y, u, unscaled=True):
    """Takes in y and prediction u"""
    if unscaled:
        # scale by setting the negative values as zero and the positive to one.
        y_inds = np.where(y>0)
        u_inds = np.where(u>0)
        y[y_inds] = 1
        u[u_inds] = 1

        y_inds = np.where(y<0)
        u_inds = np.where(u<0)
        y[y_inds] = 0
        u[u_inds] = 0

        count = 0
        for i in range(len(y)):
            if y[i]==u[i]:
                count+=1

        return count/len(y)
    else:
        count = 0
        for i in range(len(y)):
            if y[i]==u[i]:
                count+=1
        return count/len(y)



if __name__ == '__main__':
    X, y = read_in_data("defaulted_cc-clients.xls")
    method = sklearn_GDRegressor(X, y)
