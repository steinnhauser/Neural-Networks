import pandas as pd
import scipy
import sklearn
import tensorflow as tf
import numpy as np
from sklearn.linear_model import SGDRegressor
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os


def read_in_data(
    fn, headers=False, shuffle=False, seed=0, scale=True, remove_outliers=True,
        save_processed_data=True):
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

    if __name__ == "__main__":
        filename = os.path.join("../data/", fn)
    else:
        filename = os.path.join("./data", fn)

    df = pd.read_excel(filename)        # shape (30001, 25)
    dfs = np.split(df, [1, 24], axis=1) # split dataframe
    X = dfs[1].values[1:]
    y = dfs[2].values[1:]

    if remove_outliers:
        X, y = remove_categorical_outliers(X,y)
    if scale:
        X = scale_data(X)

    X = X.astype(float)
    y = y.astype(float)

    if shuffle:
        X, y = shuffle_Xy(X, y, seed)

    if save_processed_data:
        save_features_predictors(X,y)

    if headers:
        headers = df.values[0, 1:-2]  # headers of X-columns in the same order.

def remove_categorical_outliers(X,y):
    """remove categorical outliers from the X and y data"""
    # X2: Gender (1 = male; 2 = female).
    valid_mask = np.logical_and(X[:, 1] >= 1, X[:, 1] <= 2)
    y = y[valid_mask]
    X = X[valid_mask]

    # X3: Education (1 = graduate school; 2 = university;\
    #   3 = high school; 4 = others).
    valid_mask = np.logical_and(X[:, 2] >= 1, X[:, 2] <= 4)
    y = y[valid_mask]
    X = X[valid_mask]

    # X4: Marital status (1 = married; 2 = single; 3 = others)
    valid_mask = np.logical_and(X[:, 3] >= 1, X[:, 3] <= 3)
    y = y[valid_mask]
    X = X[valid_mask]

    # X6 - X11: History of past payment
    #   All take inputs from -2 to 9.
    for i in range(5, 11):
        valid_mask = np.logical_and(X[:, i] >= -2, X[:, i] <= 9)
        y = y[valid_mask]
        X = X[valid_mask]
    # there are no elements of size 9. There are 25 cases of 8.

    return X, y

def save_features_predictors(X,y):
    """Save the data as a CSV file in the ./data/ directory"""
    pwd = "./data/"
    if __name__ == "__main__":
        pwd = "." + pwd
    else:
        pass

    fn1 = os.path.join(pwd, "features.npy")
    fn2 = os.path.join(pwd, "predictors.npy")
    exists1 = os.path.isfile(fn1)
    exists2 = os.path.isfile(fn2)
    if exists1 and exists2:
        inp = input("Would you like to overwrite previous data?(y/n)")
        if str(inp) == "y" or str(inp) == "Y":
            np.save(fn1, X)
            np.save(fn2, y)
            print(f"Features X and outcomes y overwritten in files {fn1}" +
             f" and {fn2}.")
        else:
            print("Data not overwritten.")
    else:
        np.save(fn1, X)
        np.save(fn2, y)
        print(f"Features X and outcomes y saved in files {fn1} and {fn2}.")

def load_features_predictors():
    """Loads the data files of X and y from the ./data/ directory"""
    pwd = "./data/"
    if __name__ == "__main__":
        pwd = "." + pwd
    else:
        pass

    fn1 = os.path.join(pwd, "features.npy")
    fn2 = os.path.join(pwd, "predictors.npy")

    X = np.load(fn1)
    y = np.load(fn2)
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

    X_sparse = scipy.sparse.coo_matrix(X)  # a sparse matrix in coordinate format
    X, X_sparse, y = sklearn.utils.shuffle(X, X_sparse, y, random_state=seed)
    return X, y

def scale_data(X, meanzero=True, probability=False):
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
    """


    """CASES X1, X5, X12-X23: Scale large data values by indices. How these
    should be scaled is up for debate though the default is mean=0, std=1"""
    a = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    for i in a:
        if meanzero:
            # values with mean=0 and std=1:
            X[:, i] = X[:, i] - np.mean(X[:, i])
            X[:, i] = X[:, i] / np.std(X[:, i])

        elif probability:
            # values from 0 to 1:
            X[:, i] = X[:, i] - X[:, i].min()
            X[:, i] = X[:, i] / X[:, i].max()

    """CASES X6-X11: Separate categorical and continuous data. Do this first
    to avoid changing the indices for the categories lower down."""
    c = [5, 6, 7, 8, 9, 10]
    newmtxs = np.zeros(6, dtype = np.ndarray)
    i=0
    X = pd.DataFrame(X)
    for j in c:
        # 'manual' one-hot encoding:
        row1 = X[j]
        row1 = row1.apply(lambda x: 1 if x==-2. else 0)
        vec1 = row1.values
        row2 = X[j]
        row2 = row2.apply(lambda x: 1 if x==-1. else 0)
        vec2 = row2.values
        row3 = X[j]
        row3 = row3.apply(lambda x: 1 if x==0. else 0)
        vec3 = row3.values
        row4 = X[j]
        if meanzero:
            norm = np.mean([1, 2, 3, 4, 5, 6, 7, 8, 9]) # for normalization
            std  = np.std([1, 2, 3, 4, 5, 6, 7, 8, 9])
            row4 = row4.apply(lambda x: (x-norm)/std if (x>=1 and x<=9) else 0)
            vec4 = row4.values
        elif probability:
            row4 = row4.apply(lambda x: (x-1)/9 if (x>=1 and x<=9) else 0)
            vec4 = row4.values

        A = np.column_stack((vec1, vec2))
        B = np.column_stack((vec3, vec4))
        # combine the new column matrices (N,2) to a matrix of size (N,4):
        newmtxs[i] = np.append(A,B, axis=1)
        i+=1

    # need to replace the arrays from X6-X11 with these matrices:
    Xs = np.split(X, [5,11], axis=1)    # remove columns X6-X11
    E1 = Xs[0].values   # left side     dims (29601, 5)
    E2 = Xs[2].values   # right side    dims (29601, 12)

    """These matrices are all the data columns except for X6-X11. We want to
    replace these columns with the new matrices in the newmtxs list:"""
    p1 =    np.append(newmtxs[0], newmtxs[1], axis=1) # combine the matrices
    p2 =    np.append(newmtxs[2], newmtxs[3], axis=1)
    pR =    np.append(newmtxs[4], newmtxs[5], axis=1)
    pL =    np.append(p1, p2, axis=1)
    p5 =    np.append(pL, pR, axis=1)   # combine Left and Right sides
    LS =    np.append(E1, p5, axis=1)   # combine with E1 and E2
    X  =    np.append(LS, E2, axis=1)   # final scaled product

    """CASES X2, X3, X4: One-hot encoding categories. These are purely
    categorical, so the one-hot encoding is easier."""
    b = [1, 2, 3]
    b_elem = [1, 3, 2]  # no. of (additional) features from one-hot
    extra = 0           # counts the extra indices needed after additions

    for j in range(3):
        i = b[j] + extra
        series = pd.Series(X[:, i])
        dummies = pd.get_dummies(series).values  # one hot encoded
        # add array into place 'i' (sandwitch dummies between arrays)
        X = np.append(np.append(X[:, :i], \
            dummies, axis=1), X[:, i + 1 :], axis=1)
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
    clf = sklearn.linear_model.SGDRegressor(
        penalty="none",
        learning_rate="constant",
        eta0=eta0,
        max_iter=max_iter,
        fit_intercept=intercept,
        tol=tol,
    )
    clf.fit(X, y)
    return clf

def tensorflow_NNWsolver(X, y, Xt, yt):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(24, activation="tanh"),
            tf.keras.layers.Dense(16, activation="tanh"),
            tf.keras.layers.Dense(8, activation="tanh"),
            tf.keras.layers.Dense(4, activation="tanh"),
            tf.keras.layers.Dense(1, activation="sigmoid")
            # tf.keras.layers.Dropout(0.2),
        ]
    )  # try two outputs and softmax (well taylored for )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X, y, epochs=30, batch_size=100, validation_data=(Xt, yt))

    pred = model.predict(Xt)
    return pred

def assert_binary_accuracy(y, u, unscaled=True, verbose=False):
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
        u[np.where(u > 0.5)] = 1
        u[np.where(u < 0.5)] = 0

        count = 0
        for i in range(len(y)):
            if verbose:
                print(f"Output: {u[i]}, True: {y[i]}" + (\
                    " Correct!" if ypred[i]==ytrue[i] else " "))
            if y[i] == u[i]:
                count += 1
        acc = count / len(y)
        return acc

    else:
        count = 0
        for i in range(len(y)):
            if y[i] == u[i]:
                count += 1
        acc = count / len(y)
        return acc


if __name__ == "__main__":
    # read_in_data("defaulted_cc-clients.xls")
    X, y = load_features_predictors()
