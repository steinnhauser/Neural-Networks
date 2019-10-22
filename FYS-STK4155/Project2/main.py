import lib.functions as fns
import lib.logistic_regression as lgr
import lib.neural_network as nnw
import sklearn
import time
import numpy as np
# import mpi4py


def main():
    # ------------------------- Data preparation -------------------------
    sd = int(time.time())
    fn = "defaulted_cc-clients.xls"
    Xf, yf = fns.read_in_data(fn, shuffle=True, seed=sd)

    # ca. 77.88% of the data is zero. Requires resampling
    Xf, yf = fns.upsample(Xf, yf, sd)
    # Xf, yf = fns.downsample(Xf, yf, sd)

    X, Xt, y, yt = sklearn.model_selection.train_test_split(
        Xf, yf, test_size=0.5, random_state=sd
    )

    # ----------------- Classification (credit card data) ----------------
    dfrac = -1  # portion of the data to analyse. must be between 1-30000
    X, y, Xt, yt = X[:dfrac], y[:dfrac], Xt[:dfrac], yt[:dfrac]
    # Logistic Regression
    print(SGD_with_minibatches(X, y, Xt, yt, gamma=0.01,
        max_iter=1000, batch_size=100, verbose=False)) # our code
    # print(Sklearn_sgd_classifier(X, y, Xt, yt)) # comparison with sklearn
    # Artificial Neural Networks
    # print(FFNN_backpropagation(X, y, Xt, yt)) # our code
    # print(Tensorflow_neural_network(X, y, Xt, yt)) # comparison with tensorflow

    # ----------------- Regression (franke function data) ----------------

    return None


def Sklearn_sgd_classifier(X, y, Xt, yt):
    """
    Classification
    Logistic Regression
    Scikit-learns module: SGDClassifier
    """
    print("-------------------")
    solution = sklearn.linear_model.SGDClassifier(eta0=0.01, max_iter=100).fit(X, y)
    yp = solution.predict(X)
    # print(yp)
    a = fns.assert_binary_accuracy(y, yp)
    return f"Sklearn\'s SGDClassifier accuracy: {100*a:.0f} %"


def SGD_with_minibatches(X, y, Xt, yt, gamma, max_iter, batch_size, verbose=False):
    """
    Classification
    Logistic Regression
    Our own SGD with mini-batches
    (see "./lib/logistic_regressionk.py")
    """
    obj = lgr.StochasticGradientDescent(gamma, max_iter, batch_size, verbose=verbose)
    obj.fit(X, y)
    yp = obj.predict(Xt)
    # print(yp)
    a = fns.assert_binary_accuracy(y, yp)
    return f"SGD with mini-batches accuracy: {100*a:.0f} %"


def Tensorflow_neural_network(X, y, Xt, yt):
    """
    Classification
    Neural Networks
    Tensorflows module
    """
    print("-------------------")
    yp = fns.tensorflow_NNWsolver(X, y, Xt, yt)
    # print(yp)
    a = fns.assert_binary_accuracy(yt, yp)
    return f"Tensorflow NN accuracy: {100*a:.0f} %"


def FFNN_backpropagation(X, y, Xt, yt):
    """
    Classification
    Neural Networks
    Our own FFNN by using the backpropagation algorithm
    (see "./lib/neural_network.py")
    """
    n1 = nnw.Neuron(
        eta=0.1, maxiter=1, tol_bw=1e-3, cost_fn_str="xentropy", batchsize=5
    )

    n1.add_hlayer(18, activation="tanh")
    n1.add_hlayer(12, activation="tanh")
    n1.add_hlayer(6, activation="tanh")
    n1.set_outputs(y[0], activation="sigmoid")
    n1.set_inputs(X[0, :], init=True)
    n1.set_biases()
    n1.set_weights()
    n1.set_cost_fn()  # require in/outputs

    train_no, test_no = 1000, 1000
    n1.train_neuron(X, y, train_no=train_no)
    n1.test_neuron(Xt, yt, test_no=test_no, load_data=True)
    # return *make accuracy string*
    return " "


if __name__ == "__main__":
    start = time.time()
    main()
    print("-------------------")
    print(f"Completed in {time.time() - start:.2f} seconds.")

"""
Neural network slides:
https://compphysics.github.io/MachineLearning/doc/pub/NeuralNet/html/._NeuralNet-bs023.html
"""
