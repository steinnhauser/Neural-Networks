import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import scipy
import sklearn
import numpy as np



def read_in_data(fn, headers=False, shuffle=False, seed=0):
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

    if headers:
        headers = df[0,1:-2] # headers for the X-columns in the same order.
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



if __name__ == '__main__':
    X, y = read_in_data("defaulted_cc-clients.xls")
    print(X.shape)
