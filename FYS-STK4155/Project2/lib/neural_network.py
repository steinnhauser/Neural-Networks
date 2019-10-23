import numpy as np
import lib.functions as fns


class Neuron:
    """
    Neuron class. Initialize before adding hidden layers and outputs.

    Parameters:
    -----------
        eta : float
            Learning rate for the bias and weight differences.
        n_hlayers : float, default 0
            Number of hidden layers in the network. Initialize as 0.
        nodes : int list, default []
            List of the number of nodes per layer. Initialize as [].
        act_fn : lambda list, default []
            List of the activation functions for each layer. Initialize as []
        act_der : lambda list, default []
            List of the activation function derivatives. Initialize as []
        biases_str : str, default 'random'
            Initialization of the biases. Takes 'zeros' and 'small' as well.
        weights_str : str, default 'random'
            Initialization of the weights. Takes 'zeros' and 'small' as well.
        cost_fn_str : str, default 'MSE'
            Initialize of the loss function. Takes 'xentropy' and 'softmax' too.
        datasets_sampled : int, default 0
            Counts how many datasets have been applied to the training process
        verbose : bool, default False
            Prints unnessecary indications of training and testing
        tol_bw : float, default 1e-5
            Tolerance which dicates when the network is properly trained
        maxiter : int, default 20
            Maximum amount of iterations before moving on

    Functions:
    ----------
        add_hlayer
        set_activation
        set_biases
        set_weights
        set_cost_fn
        set_inputs_outputs
        fb_propogation
        feedforward
        backpropogate
        output_func
        train_neuron
        test_neuron
        save_data
        load_data
        assert_accuracy
    """

    def __init__(
        self,
        eta=0.1,
        biases_str="random",
        weights_str="random",
        cost_fn_str="MSE",
        datasets_sampled=0,
        verbose=False,
        tol_bw=1e-5,
        maxiter=1,
        batchsize=10,
    ):

        self.eta = eta  # learning rate
        self.n_hlayers = 0  # no. of hidden layers

        self.nodes = []  # list of nodes for each layer
        self.act_fn = []  # list of activation functions
        self.act_der = []  # list of activation function derivatives

        self.biases_str = biases_str
        self.weights_str = weights_str
        self.cost_fn_str = cost_fn_str

        self.outputs_set = False  # if the output is set
        self.inputs_set = False  # if the input is set

        self.datasets_sampled = datasets_sampled  # sets the network has been trained on
        self.verbose = verbose  # print unnessecary indications
        self.tol_bw = tol_bw  # bias and weight diff tolerance
        self.maxiter = maxiter  # max number of f/b propogation iterations
        self.batchsize = batchsize  # size of batches

    def add_hlayer(self, n_nodes, activation):
        """Function to add a hidden layer to the network. Sets activation for
        the layer and counts the number of hidden layers/nodes automatically"""
        self.set_activation(activation)
        self.nodes += [n_nodes]
        self.n_hlayers += 1

    def set_activation(self, act_str):
        """Add an activation function of the network activation function list"""
        if act_str == "sigmoid":
            sigmoid = np.vectorize(lambda z: 1.0 / (1 + np.exp(-z)))
            self.act_fn += [sigmoid]
            self.act_der += [np.vectorize(lambda z: sigmoid(z) * (1 - sigmoid(z)))]
        elif act_str == "tanh":
            self.act_fn += [np.vectorize(lambda z: np.tanh(z))]
            self.act_der += [np.vectorize(lambda z: 1 - np.tanh(z) ** 2)]
        elif act_str == "relu":
            self.act_fn += [np.vectorize(lambda z: 0 if z < 0.0 else z)]
            self.act_der += [np.vectorize(lambda z: 0 if z < 0.0 else 1)]
        elif act_str == "softmax":
            self.act_fn += [np.vectorize(lambda u: np.exp(u) / np.sum(np.exp(u)))]
            self.act_der += [
                np.vectorize(lambda u: self.cost_fn(u) * (1 - self.cost_fn(u)))
            ]
        else:
            raise SyntaxError(
                "Activation function must be 'sigmoid'\
                or 'tanh' or 'relu' or 'softmax'"
            )

    def set_biases(self):
        """Set the biases of the network."""
        if self.biases_str == "zeros":
            self.biases = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
            for i in range(self.n_hlayers + 1):
                self.biases[i] = np.zeros(self.nodes[i + 1])
        elif self.biases_str == "random":
            self.biases = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
            for i in range(0, self.n_hlayers + 1):
                self.biases[i] = np.random.rand(self.nodes[i + 1]) - 0.5
        elif self.weights_str == "small":
            self.biases = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
            for i in range(0, self.n_hlayers + 1):
                self.biases[i] = np.zeros(self.nodes[i + 1]) + 0.1
        elif self.biases_str == "custom":
            pass
        else:
            raise SyntaxError("Biases must be 'zeros', 'random' or 'small'.")

    def set_weights(self):
        """Set the weights of the network. The weights matrix should
        have dimensions (next nodes x prev nodes)"""
        if self.weights_str == "zeros":
            self.weights = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
            for i in range(self.n_hlayers + 1):
                self.weights[i] = np.zeros((self.nodes[i], self.nodes[i + 1]))
        elif self.weights_str == "random":
            self.weights = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
            for i in range(self.n_hlayers + 1):
                self.weights[i] = np.random.rand(self.nodes[i], self.nodes[i + 1]) - 0.5
                # should have dim (prev layer x next layer)
        elif self.weights_str == "small":
            self.weights = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
            for i in range(self.n_hlayers + 1):
                self.weights[i] = np.zeros(self.nodes[i], self.nodes[i + 1]) + 0.1
        elif self.weights_str == "custom":
            pass
        else:
            raise SyntaxError("Weights must be 'zeros', 'random' or 'small'.")

    def set_cost_fn(self):
        """Set the cost function"""
        if self.cost_fn_str == "MSE":
            self.cost_fn = lambda u: 0.5 * (self.y - u) ** 2
            self.cost_der = lambda u: np.vectorize(-(self.y - u))
        elif self.cost_fn_str == "xentropy":
            self.cost_fn = lambda u: -self.y * np.log(u)
            self.cost_der = lambda u: u - self.y
        else:
            raise SyntaxError("Cost function must be 'MSE' or 'xentropy'")

    def set_outputs(self, y, activation):
        """Sets the outputs y (shape batch x 1). Arranges nodes respectively.
        The function assumes that the network is 'many-to-one'."""

        self.y = y
        if not self.outputs_set:
            self.nodes = self.nodes + [1]  # one output
            self.set_activation(activation)
            self.outputs_set = True

    def set_inputs(self, X, init=False):
        """Sets the inputs X (shape batch x p)"""
        self.X = X
        if not init:
            if self.batchsize != X.shape[0]:
                raise ValueError("Error, shape of X does not match batchsize.")

        if not self.inputs_set and init:
            self.nodes = [X.shape[0]] + self.nodes
            self.inputs_set = True

    def fb_propogation(self):
        """
        Main program which conducts the forward and backwards
        propogation through the neural network for training.
        """
        self.err_bw = self.tol_bw + 1  # initialize this for the first loop
        self.datasets_sampled += 1  # counts how many instances.

        self.feedforward()
        self.backpropogate()

    def feedforward(self):
        """Feed the network forward and save z and a"""
        # Initialize the input 'z' matrix: should be (batchsize x nhlayers+1)
        self.z = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
        # Initialize the output 'a' matrix:
        self.a = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)

        # initialize first ('zeroth') layer
        self.z[0] = self.X @ self.weights[0] + self.biases[0]
        self.a[0] = self.act_fn[0](self.z[0])

        for l in range(1, self.n_hlayers + 1):
            self.z[l] = self.a[l - 1] @ self.weights[l] + self.biases[l]
            self.a[l] = self.act_fn[l](self.z[l])

        self.output = self.a[-1]

    def backpropogate(self):
        """Backpropogate and adjust the weights and biases"""
        # Initialize error vector delta (one for each layer except input)
        d = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)

        # set up arrays for the differences of bias and weights:
        diff_b = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
        diff_w = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)

        # Last diff element l=L
        der1 = self.act_der[-1](self.z[-1])
        der2 = -2 * (self.y.reshape(-1, 1) - self.output[0])
        d[-1] = der1 * der2

        diff_b[-1] = np.sum(d[-1]).reshape(-1)
        diff_w[-1] = self.a[-2].T @ d[-1]

        for l in range(self.n_hlayers - 1, 0, -1):
            d[l] = (d[l + 1] @ self.weights[l + 1].T) + self.act_der[l](self.z[l])
            diff_b[l] = np.sum(d[l], axis=0)
            diff_w[l] = self.a[l - 1].T @ d[l]

        l -= 1  # last layer, where a[0] = X
        d[l] = (d[l + 1] @ self.weights[l + 1].T) + self.act_der[l](self.z[l])
        diff_b[l] = np.sum(d[l], axis=0)
        diff_w[l] = self.X.T @ d[l]

        print("-----------")
        self.err_bw = 0
        for i in range(self.n_hlayers + 1):
            self.biases[i] -= self.eta * diff_b[i]
            self.weights[i] -= self.eta * diff_w[i]
            self.err_bw += abs(np.sum(diff_b[i])) + abs(np.sum(diff_w[i]))
        print(self.err_bw)
        # The parameter eta is the learning parameter discussed in connection
        # with the gradient descent methods. Here it is convenient to use
        # stochastic gradient descent with mini-batches with an outer loop
        # that steps through the multiple epochs of training. Try to parallize?

    def output_func(self):
        """Function to produce an output for a given input"""
        iter = self.act_fn[0](self.X @ self.weights[0] + self.biases[0])
        for l in range(1, self.n_hlayers + 1):
            iter = self.act_fn[l](iter @ self.weights[l] + self.biases[l])
        self.output = iter

    def train_neuron(self, X, y, train_no=100):
        """Function to train the netorks weights and biases"""
        done = 0
        iterations = int(train_no / self.batchsize)
        for i in range(iterations):
            s = i * self.batchsize  # start
            e = s + self.batchsize  # end
            self.set_inputs(X[s:e, :])
            self.set_outputs(y[s:e], activation="sigmoid")
            self.fb_propogation()
            if str(self.output) == "[nan]":
                print("Something went wrong. Output is 'nan'...")
                break
            if self.err_bw < self.tol_bw and not done:
                print(f"Network is trained up to tolerance after {s} sets.")
                done += 1
        print("--------------")
        print("Network Trained.")
        self.save_data()

    def test_neuron(self, X, y, test_no=100, load_data=False):
        """Function the test the networks capabilities"""
        if self.verbose:
            print("-------------")
            print("Testing:")
        if load_data:
            self.load_data()
        correct = 0
        iterations = int(test_no / self.batchsize)
        for i in range(iterations):
            s = i * self.batchsize  # start
            e = s + self.batchsize  # end
            self.set_inputs(X[s:e, :])
            self.set_outputs(y[s:e], activation="sigmoid")
            print(self.X.shape)
            print(self.weights[0].shape)
            print(self.biases[0].shape)
            self.feedforward()  # feed the network forward once using W and b.
            # self.output_func() # produce prediction
            if (self.output > 0.5 and self.y == 1) or (
                self.output < 0.5 and self.y == 0
            ):
                correct += 1

            self.verbose = True
            if self.verbose:
                print(f"Output:\t{self.output}\tTrue:\t{self.y}")
        print(f"Network had an accuracy of {correct/test_no*100:.2f} %")

    def save_data(self):
        """Function to save the weights and biases for a network"""
        answer = input("Would you like to save the weight and bias data? (y/n)")
        pwd = "bin/"
        if answer == "y" or answer == "Y":
            ans = input("Overwrite previous data? (y/n)")
            if ans == "y" or ans == "Y":
                outfile1 = pwd + "weight_data.npy"
                outfile2 = pwd + "bias_data.npy"
                np.save(outfile1, self.weights, allow_pickle=True)
                np.save(outfile2, self.biases, allow_pickle=True)
                print(f"Data overwritten in files {outfile1} and {outfile2}.")
            elif ans == "n" or ans == "N":
                fn = str(input("Input new file name:"))
                outfile1 = pwd + fn + "1.npy"
                outfile2 = pwd + fn + "2.npy"
                np.save(outfile1, self.weights, allow_pickle=True)
                np.save(outfile2, self.biases, allow_pickle=True)
                print(f"Weights saved in {outfile1} and bias in {outfile2}.")
            else:
                raise SyntaxError("(y/n) was not input. Aborting.")
        else:
            print("Data not saved.")

    def load_data(self, fn1="weight_data.npy", fn2="bias_data.npy"):
        """Function to load the weights and biases for a network"""
        pwd = "bin/"
        self.weights = np.load(pwd + fn1, allow_pickle=True)
        self.biases = np.load(pwd + fn2, allow_pickle=True)
        print("Data loaded.")

    def assert_accuracy(self, X, y, test_sample=100):
        """Function to assert the binary accuracy of a network"""
        results = np.zeros(test_sample)
        for i in range(test_sample):
            self.X = X[i, :]
            self.feedforward()
            output = self.output_func()
            results[i] = output

        # results = (np.random.random(test_sample))*0.7/0.5 # test this as results.
        self.acc = fns.assert_binary_accuracy(y[:test_sample], results)
        print(f"Network produced an accuracy of {100*self.acc:.2f}%")


if __name__ == "__main__":
    pass
