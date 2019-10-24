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
    # fns.read_in_data(fn)  # to preprocess and save features X and outcomes y
    Xf, yf = fns.load_features_predictors() # load preprocessed data

    # ca. 77.88% of the data is zero. Requires resampling
    Xf, yf = fns.upsample(Xf, yf, sd)
    # Xf, yf = fns.downsample(Xf, yf, sd)

    X, Xt, y, yt = sklearn.model_selection.train_test_split(
        Xf, yf, test_size=0.1, random_state=sd, stratify=yf
    )

    # ----------------- Classification (credit card data) ----------------
    dfrac = -1  # portion of the data to analyse. must be between 1-30000
    X, y, Xt, yt = X[:dfrac], y[:dfrac], Xt[:dfrac], yt[:dfrac]
    # Logistic Regression
    # print(SGD_with_minibatches(X, y, Xt, yt, gamma=0.01,
        # max_iter=1000, batch_size=100, verbose=False)) # our code
    # print(Sklearn_sgd_classifier(X, y, Xt, yt)) # comparison with sklearn
    # Artificial Neural Networks
    print(FFNN_backpropagation(X, y, Xt, yt)) # our code
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
        eta=0.1, maxiter=1, tol_bw=1e-3, cost_fn_str="MSE", batchsize=10
    )

    n1.add_hlayer(60, activation="tanh")
    n1.add_hlayer(50, activation="tanh")
    n1.add_hlayer(40, activation="tanh")
    n1.add_hlayer(30, activation="tanh")
    n1.add_hlayer(20, activation="tanh")
    n1.add_hlayer(10, activation="tanh")
    n1.add_hlayer(6, activation="tanh")
    n1.set_outputs(y[0], activation="sigmoid")
    n1.set_inputs(X[0, :], init=True)
    n1.set_biases()
    n1.set_weights()
    n1.set_cost_fn()  # require in/outputs

    # n1.train_neuron(X, y, epochs=5000)
    customfn = "sixty_tanh_b10"  # custom saved weights and biases
    n1.test_neuron(Xt, yt, load_data=True, cfn=customfn)
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


"""
Accomplished:
Xentropy:
    Network had an accuracy of 62.13 %
MSE:
    Network had an accuracy of 63.30 %

Using:

"""

"""
Record:
68.74% accuracy using:
    Nodes   [40, 30, 18, 12, 6, 1],
    ActFn   [t,  t,  t,  t,  t, s],
MSE, and 10% testing data. 1000 epochs, batch size 10
"""
