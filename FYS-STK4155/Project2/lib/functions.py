import pandas as pd
import scipy
import sklearn
import tensorflow as tf
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

def sklearn_GDRegressor(X, y, intercept=False, eta0=0.01, max_iter=50, tol=1e-3):
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
        'constant', eta0=eta0, max_iter=max_iter, fit_intercept=intercept,\
        tol=tol)
    clf.fit(X,y)
    return clf

def tensorflow_NNWsolver(X, y, Xt, yt, nodes=[24, 18, 12, 6, 1],\
    act_fn_str = "sigmoid"):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(i, activation = act_fn_str) for i in nodes
        ]
    )
    model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    model.fit(
        X, y,
        epochs = 1,
        batch_size = 1,
        validation_data = (Xt, yt)
    )

    pred = model.predict(Xt)

def assert_binary_accuracy(y, u, unscaled=True):
    """
    Takes in testing data y and prediction u and
    calculates the accuracy of the prediction.

    Parameters:
    -----------
    y : vec
        (N x 1) vector of testing data.
    u : vec
        (N x 1) vector of prediction data.
    unscaled : bool, default True
        Determines whether the data should be scaled or not.

    Returns:
    --------
    acc : float
        Accuracy of the model in relation to the testing data (between 0 and 1).
    """
    if unscaled:
        # declare y<0.5 to be 0 and y>0.5 to be 1.
        u[np.where(u>0.5)] = 1
        u[np.where(u<0.5)] = 0

        count = 0
        for i in range(len(y)):
            if y[i]==u[i]:
                count+=1
        acc = count/len(y)
        return acc

    else:
        count = 0
        for i in range(len(y)):
            if y[i]==u[i]:
                count+=1
        acc = count/len(y)
        return acc


if __name__ == '__main__':
    X, y = read_in_data("defaulted_cc-clients.xls")
    method = sklearn_GDRegressor(X, y)
