import lib.functions as fns
import lib.logistic_regression as lgr
import sklearn
import time
import numpy as np


def main():
    sd = int(time.time())
    fn = "defaulted_cc-clients.xls" # data filename
    X, y = fns.read_in_data(fn, shuffle=True, seed = sd, scale=True)
    ymean = np.mean(y)
    ystd = np.std(y)
    SKL(X,y)
    GDS(X,y)
    # CGM(X,y)

def SKL(X, y):
    """Sklearn method"""
    print("-------------------")
    solution = fns.sklearn_GDRegressor(X, y).coef_
    y_pred = X @ solution
    print("Sklearn prediction vs. true values:")
    print(y_pred)
    print(y)
    a = fns.assert_binary_accuracy(y, y_pred)
    print(f"SKL had accuracy of {100*a:.0f} %")

def GDS(X, y):
    """Gradient Descent solver"""
    print("-------------------")
    solution = lgr.gradient_descent_solver(X, y, random_state_x0=True)
    y_pred = X @ solution
    # y_pred = sklearn.preprocessing.scale(y_pred)
    print("Customized GD vs. true values:")
    print(y_pred) # scaling the end works p w
    print(y)
    a = fns.assert_binary_accuracy(y, y_pred)
    print(f"GDS had accuracy of {100*a:.0f} %")

def CGM(X, y):
    """Conjugate Gradient method"""
    print("-------------------")
    solution = lgr.CGMethod(X, y, random_state_x0=True)
    y_pred = X @ solution
    print("Customized GD vs. true values:")
    print(y_pred)
    print(y)
    a = fns.assert_binary_accuracy(y, y_pred)
    print(f"CGM had accuracy of {100*a:.0f} %")


if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Completed in {time.time() - start:.2f} seconds.")

""" SKL Printout:
GD reached max iteration.
Customized GD vs. true values:
[10.89857803 -1.2031399   9.34469721 ... -1.68943361  2.82401563
 -3.97601969]
[ 1.87637834 -0.53294156  1.87637834 ... -0.53294156  1.87637834
 -0.53294156]
Completed in 38.61 seconds.
"""
