import lib.functions as fns
import lib.logistic_regression as lgr
import sklearn
import time
import numpy as np


def main():
    sd = int(time.time())
    fn = "defaulted_cc-clients.xls"
    X, y = fns.read_in_data(fn, shuffle=True, seed = sd, scale=True)
    SKL(X,y)
    GDS(X,y)
    # CGM(X,y)

def SKL(X, y, yscale=True):
    """Sklearn method"""
    print("-------------------")
    solution = fns.sklearn_GDRegressor(X, y).coef_
    y_pred = X @ solution
    if yscale:
        print("y is being scaled.")
        y_pred = sklearn.preprocessing.scale(y_pred)
    a = fns.assert_binary_accuracy(y, y_pred)
    print(f"SKL had accuracy of {100*a:.0f} %")

def GDS(X, y, yscale=True):
    """Gradient Descent solver"""
    print("-------------------")
    solution = lgr.gradient_descent_solver(X, y, random_state_x0=True)
    y_pred = X @ solution
    if yscale:
        print("y is being scaled.")
        y_pred = sklearn.preprocessing.scale(y_pred)
    a = fns.assert_binary_accuracy(y, y_pred)
    print(f"GDS had accuracy of {100*a:.0f} %")

def CGM(X, y):
    """Conjugate Gradient method"""
    print("-------------------")
    solution = lgr.CGMethod(X, y, random_state_x0=True)
    y_pred = X @ solution
    a = fns.assert_binary_accuracy(y, y_pred)
    print(f"CGM had accuracy of {100*a:.0f} %")


if __name__ == '__main__':
    start = time.time()
    main()
    print("-------------------")
    print(f"Completed in {time.time() - start:.2f} seconds.")

"""
steinn@SHM-PC:~/Desktop/Neural-Networks/FYS-STK4155/Project2$ python3 -W ignore main.py
-------------------
SKL had accuracy of 99 %
-------------------
GD reached tolerance.
y is being scaled.
GDS had accuracy of 100 %
-------------------
Completed in 3.67 seconds.
"""
