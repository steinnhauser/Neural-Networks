import lib.functions as fns
import lib.logistic_regression as lgr
import sklearn
import time
import numpy as np


def main():
    sd = int(time.time())
    fn = "defaulted_cc-clients.xls"
    X, y = fns.read_in_data(fn, shuffle=True, seed = sd, scale=True)
    X, Xt, y, yt = sklearn.model_selection.train_test_split(X, y, \
            test_size=0.7, random_state=sd)
    SKL(X, y, Xt, yt)
    GDS(X, y, Xt, yt)
    # CGM(X, y, Xt, yt)

    neuron()

def SKL(X, y, Xt, yt):
    """Sklearn method"""
    print("-------------------")
    solution = fns.sklearn_GDRegressor(X, y).coef_
    yp = Xt @ solution # prediction
    a = fns.assert_binary_accuracy(yt, yp)
    print(f"SKL had accuracy of {100*a:.0f} %")

def GDS(X, y, Xt, yt):
    """Gradient Descent solver"""
    print("-------------------")
    solution = lgr.gradient_descent_solver(X, y, x0=100)
    yp = Xt @ solution # prediction
    a = fns.assert_binary_accuracy(yt, yp)
    print(f"GDS had accuracy of {100*a:.0f} %")

def CGM(X, y, Xt, yt):
    """Conjugate Gradient method"""
    print("-------------------")
    solution = lgr.CGMethod(X, y, random_state_x0=True)
    yp = Xt @ solution
    a = fns.assert_binary_accuracy(yt, yp)
    print(f"CGM had accuracy of {100*a:.0f} %")

def neuron():
    f = np.vectorize(lambda z: 1./(1+np.exp(z))) # activation function.


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

"""
Neural network slides:
https://compphysics.github.io/MachineLearning/doc/pub/NeuralNet/html/._NeuralNet-bs023.html
"""
