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
    Xf, yf, headers = \
        fns.read_in_data(fn, shuffle=True, seed = sd, headers=True)

    # ca. 77.88% of the data is zero. Requires resampling
    # Xf, yf = fns.upsample(Xf, yf, sd)
    Xf, yf = fns.downsample(Xf, yf, sd)

    X, Xt, y, yt = sklearn.model_selection.train_test_split(Xf, yf, \
            test_size=0.5, random_state=sd) # stratify=yf

    dfrac = 10000 # portion of the data to analyse. must be between 1-30000
    X, y, Xt, yt = X[:dfrac], y[:dfrac], Xt[:dfrac], yt[:dfrac]

    # SKL(X, y, Xt, yt)
    # GDS(X, y, Xt, yt)

    TFL(X, y, Xt, yt)
    # NNW(X, y, Xt, yt)

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
    print("-------------------")
    yp = fns.tensorflow_NNWsolver(X, y, Xt, yt)
    print(yp)
    a = fns.assert_binary_accuracy(yt, yp)
    print(f"TFL had accuracy of {100*a:.0f} %")

def NNW(X, y, Xt, yt):
    n1 = nnw.Neuron(eta=1, maxiter=100, tol_bw=1e-3,
        verbose=True, cost_fn_str="xentropy")

    n1.add_hlayer(18, activation='relu')
    n1.add_hlayer(12, activation='relu')
    n1.add_hlayer(6, activation='relu')
    n1.set_inputs_outputs(X[0,:], y[0], activation='sigmoid')
    n1.set_biases(); n1.set_weights(); n1.set_cost_fn()

    train_no, test_no = 200, 20
    n1.train_neuron(X, y, train_no=train_no)
    n1.test_neuron(Xt, yt, test_no=test_no, load_data=True)

if __name__ == '__main__':
    start = time.time()
    main()
    print("-------------------")
    print(f"Completed in {time.time() - start:.2f} seconds.")

"""
Neural network slides:
https://compphysics.github.io/MachineLearning/doc/pub/NeuralNet/html/._NeuralNet-bs023.html
"""
