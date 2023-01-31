import sklearn
from sklearn.linear_model import RidgeCV as RCV
from sklearn.linear_model import LassoCV as LCV
import pandas as pd
import numpy as np
import time


def main():
    data = pd.read_csv("data_E1.csv").values # 110 rows x 22285 columns
    GSMv = data[:,0] # GSM table names.
    CADi = data[:,1] # known Duke CAD indices.

    # First two columns are 'Unnamed: 0', and 'CADi'.
    # These are not included in the regression.

    Xdata = np.array((data[:, 2:])) # shape (110, 22283)
    ydata = np.array(CADi)          # shape (110, )

    # scale the data properly (work using fit_intercept=False)
    # ymean = np.mean(ydata) can be optionally saved for rescaling.
    X = sklearn.preprocessing.scale(Xdata)
    y = sklearn.preprocessing.scale(ydata)

    X_train, X_test, y_train, y_test\
        = sklearn.model_selection.train_test_split(X, y,\
         test_size=0.2, random_state=1)

    ridge_model         = Ridge_reg(X_train, y_train)
    ridge_prediction    = ridge_model.predict(X_test)
    R2_Ridge    = ridge_model.score(X_test, y_test)
    MSE_Ridge   = np.sum((y_test-ridge_prediction)**2)

    lasso_model         = Lasso_reg(X_train, y_train)
    lasso_prediction    = lasso_model.predict(X_test)
    R2_Lasso    = lasso_model.score(X_test, y_test)
    MSE_Lasso   = np.sum((y_test-lasso_prediction)**2)

    print(f"The Ridge model had:\n\tR2={R2_Ridge:.2f}\n\tMSE={MSE_Ridge:.2f}")
    print(f"The Lasso model had:\n\tR2={R2_Lasso:.2f}\n\tMSE={MSE_Lasso:.2f}")


def Ridge_reg(X, y):
    model = RCV(fit_intercept=False, cv=5).fit(X, y)
    return model

def Lasso_reg(X, y):
    model = LCV(n_jobs=-1, fit_intercept=False, cv=5).fit(X, y)
    return model

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Completed in {time.time() - start:.2f} seconds.")

"""
steinn@SHM-PC:~/Desktop/STK-IN4300/P1$ python3 Ex1.py
The Ridge model had:
	R2=0.21
	MSE=13.11
The Lasso model had:
	R2=0.06
	MSE=15.61
Completed in 32.07 seconds.
"""
