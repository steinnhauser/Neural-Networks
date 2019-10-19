import lib.functions as fns
import lib.logistic_regression as lgr
import lib.neural_network as nnw
import sklearn
import time
import numpy as np
import mpi4py


def main():
    sd = int(time.time())
    fn = "defaulted_cc-clients.xls"
    Xf, yf = fns.read_in_data(fn, shuffle=True, seed = sd)
    # ca. 77.88% of the data is zero.

    X, Xt, y, yt = sklearn.model_selection.train_test_split(Xf, yf, \
            test_size=0.5, random_state=sd, stratify=yf)

    dfrac = 1000 # portion of the data to analyse. must be between 1-30000
    X, y, Xt, yt = X[:dfrac], y[:dfrac], Xt[:dfrac], yt[:dfrac]

    # SKL(X, y, Xt, yt)
    # GDS(X, y, Xt, yt)

    # TFL(X, y, Xt, yt)
    NNW(X, y, Xt, yt)

def SKL(X, y, Xt, yt, regress=False):
    """Sklearn method"""
    print("-------------------")
    if regress:
        print("Classification chosen")
        solution = sklearn.linear_model.SGDClassifier(eta0=0.01, \
            max_iter=100).fit(X,y)
        yp = solution.predict(X)
        print(yp)
    else:
        print("Regression chosen")
        solution = fns.sklearn_GDRegressor(X, y,\
        eta0=0.01, max_iter=100, tol=1e-3).coef_
        yp = Xt @ solution # prediction
    a = fns.assert_binary_accuracy(y, yp)
    print(f"SKL had accuracy of {100*a:.0f} %")

def GDS(X, y, Xt, yt):
    """Gradient Descent solver"""
    print("-------------------")
    obj = lgr.GradientDescent()
    obj.solve(X, y)
    obj.predict(Xt, yt)

def TFL(X, y, Xt, yt):
    fns.tensorflow_NNWsolver(X, y, Xt, yt)

def NNW(X, y, Xt, yt):
    n1 = nnw.Neuron(eta=1, maxiter=100, tol_bw=1e-3, act_str="sigmoid",\
        verbose=True, cost_fn_str="xentropy")
    train_no, test_no = 200, 20
    # n1.set_inputs_outputs(X, y)
    # n1.train_neuron(X, y, train_no=train_no)
    n1.test_neuron(Xt, yt, test_no=test_no, load_data=True)

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
steinn@SHM-PC:~/Desktop/Neural-Networks/FYS-STK4155/Project2$ python3 -W ignore main.py
NNW had accuracy of 93 %
-------------------
Completed in 2.43 seconds.
"""

"""
Neural network slides:
https://compphysics.github.io/MachineLearning/doc/pub/NeuralNet/html/._NeuralNet-bs023.html
"""
