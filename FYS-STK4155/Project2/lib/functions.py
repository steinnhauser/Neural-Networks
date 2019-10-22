import pandas as pd
import scipy
import sklearn
import tensorflow as tf
import numpy as np
from sklearn.linear_model import SGDRegressor
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# column transformer for one hot useage
# scikit learn one hot

def read_in_data(fn, headers=False, shuffle=False, seed=0, scale=True,\
            remove_outliers=True):
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
    dfs = np.split(df, [1,24], axis=1)
    X = dfs[1].values[1:]
    y = dfs[2].values[1:]

    if remove_outliers:
        """remove categorical outliers"""
        # X2: Gender (1 = male; 2 = female).
        valid_mask = np.logical_and(X[:,1] >= 1, X[:,1] <= 2)
        y = y[valid_mask]
        X = X[valid_mask]

        # X3: Education (1 = graduate school; 2 = university;\
        #   3 = high school; 4 = others).
        valid_mask = np.logical_and(X[:,2] >= 1, X[:,2] <= 4)
        y = y[valid_mask]
        X = X[valid_mask]

        # X4: Marital status (1 = married; 2 = single; 3 = others)
        valid_mask = np.logical_and(X[:,3] >= 1, X[:,3] <= 3)
        y = y[valid_mask]
        X = X[valid_mask]

    if scale:
        X = scale_data(X)

    X = X.astype(float)
    y = y.astype(float)

    if shuffle:
        X, y = shuffle_Xy(X, y, seed)

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

def scale_data(X):
    """
    Function to scale the columns of X.
    Columns which have large datapoints:
        X1: LIMIT_B
        X5: AGE
        X12-X23: BILLS.
    Columns which require one-hot encoders:
        X2: SEX (1 or 2)
        X3: EDUCATION (1, 2, 3 or 4)
        X4: MARRIAGE (1, 2 or 3)
    Other:
        X6 - X11: values -2 to 9.
            1-9 are months later.
            0   is 'customer payed minimum due amount but not entire balance'
            -1  is 'Balance payed in full, but account has a positive Balance
                    at the end of period.'
            -2  is 'Balance payed in full and no transactions in this period'
                    (inactive)
    The other data is not necessary to scale.
    """

    # Scale large data values by indices
    a = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    for i in a:
        X[:,i] = X[:,i] - X[:,i].min()
        X[:,i] = X[:,i] / X[:,i].max()

    # One-hot encoding values by indices
    b = [1, 2, 3]
    b_elem = [1, 3, 2] # no. of (additional) features from one-hot
    extra = 0 # counts the extra indices needed after additions

    for j in range(3):
        i=b[j]+extra
        series = pd.Series(X[:,i])
        dummies = pd.get_dummies(series).values # one hot encoded
        # add array into place 'i' (sandwitch dummies between arrays)
        X = np.append(np.append(X[:,:i], dummies, axis=1), X[:,i+1:], axis=1)
        # adding columns changes the 'i' indices we need.
        extra += b_elem[j]

    return X

def upsample(X, y, seed):
    """Function which generates copies of the minority class"""
    ros = RandomOverSampler(random_state=seed)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

def downsample(X, y, seed):
    """Function which removes samples of the majority class"""
    rus = RandomUnderSampler(random_state=seed)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

def sklearn_GDRegressor(X, y, intercept=False, eta0=0.1, max_iter=50, tol=1e-3):
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

def tensorflow_NNWsolver(X, y, Xt, yt):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(24, activation='tanh'),
            tf.keras.layers.Dense(16, activation='tanh'),
            tf.keras.layers.Dense(8, activation='tanh'),
            tf.keras.layers.Dense(4, activation='tanh'),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
            # tf.keras.layers.Dropout(0.2),
        ]
    ) # try two outputs and softmax (well taylored for )
    model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    model.fit(
        X, y,
        epochs = 30,
        batch_size =100,
        validation_data = (Xt, yt)
    )

    pred = model.predict(Xt)
    return pred

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
            print(f"Output: {u[i]}, True: {y[i]}")
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
