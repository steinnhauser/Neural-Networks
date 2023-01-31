import numpy as np
import lib.functions as fns
import scipy.special as sps


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
        biases_str="small",
        weights_str="small",
        cost_fn_str="MSE",
        verbose=False,
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

        self.datasets_sampled = 0 # sets the network has been trained on
        self.verbose = verbose  # print unnessecary indications
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
        elif act_str == "relu6":
            # z for 0<z<6, 0 for z<0, 6 for z>6.
            self.act_fn += \
            [np.vectorize(lambda z: 0 if z < 0.0 else (z if (0<z<6) else 6))]
            self.act_der += \
            [np.vectorize(lambda z: 0 if z < 0.0 else (1 if (0<z<6) else 0))]
        elif act_str == "softmax":
            self.act_fn += [
                np.vectorize(lambda u: np.exp(u) / np.sum(np.exp(u)))
            ]
            self.act_der += [
                np.vectorize(lambda u: self.cost_fn(u) * (1 - self.cost_fn(u)))
            ]
        elif act_str == "softsign":
            self.act_fn += [np.vectorize(lambda u: u/(1+np.abs(u)))]
            self.act_der += [
                np.vectorize(lambda u: 1./(1+np.abs(u))**2)
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
        elif self.biases_str == "small":
            self.biases = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
            for i in range(0, self.n_hlayers + 1):
                self.biases[i] = np.zeros(self.nodes[i + 1]) + 0.01
        elif self.biases_str == "xavier":
            # Xavier initializes with biases=0 for all layers.
            self.biases = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
            for i in range(self.n_hlayers + 1):
                self.biases[i] = np.zeros(self.nodes[i + 1])
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
                self.weights[i] = \
                    np.random.rand(self.nodes[i], self.nodes[i + 1]) - 0.5
                # should have dim (prev layer x next layer)
        elif self.weights_str == "small":
            self.weights = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
            for i in range(self.n_hlayers + 1):
                self.weights[i] = \
                    np.zeros((self.nodes[i], self.nodes[i + 1])) + 0.01
        elif self.weights_str == "xavier":
            # Initialize as Xavier does in his paper:
            self.weights = np.zeros(self.n_hlayers + 1, dtype=np.ndarray)
            for i in range(self.n_hlayers + 1):
                self.weights[i] = \
                    np.random.normal(
                        loc=0.0,
                        scale=np.sqrt(2./(self.nodes[i]+self.nodes[i+1])),
                        size=(self.nodes[i], self.nodes[i + 1])
                    )
        elif self.weights_str == "custom":
            pass
        else:
            raise SyntaxError("Weights must be 'zeros', 'random' or 'small'.")

    def regularization(self):
        if self.reg_str == " ":
            return 0
        elif self.reg_str == "l1":
            a = 0
            for i in range(self.n_hlayers+1):
                a+= np.sum(abs(self.weights[i]))/(2*self.features)
            return self.hyperp*a
        elif self.reg_str == "l2":
            a = 0
            for i in range(self.n_hlayers+1):
                a+= np.sum(self.weights[i])**2/(2*self.features)
            return self.hyperp*a

        else:
            raise SyntaxError("Regularization must be l1, l2.")

    def set_cost_fn(self, reg_str = ' ', hyperp=0.1):
        """Set the cost function. Also assign a regularization to the same
        cost function. Regularizations are between l1 and l2, where the weights
        are summed over using a hyperparameter."""
        self.hyperp = hyperp
        self.reg_str = reg_str
        if self.cost_fn_str == "MSE":
            self.cost_fn = lambda u: \
                (1./2*self.features) * np.sum(self.y-u)**2 +\
                    self.regularization()
            self.cost_der = lambda u: \
                np.vectorize(-(self.y - u))
        elif self.cost_fn_str == "xentropy":
            self.cost_fn = lambda u: -np.sum(sps.xlogy(self.y, u) +\
                sps.xlogy(1-self.y, 1-u))/self.y.shape[0] #+ self.regularization()
            self.cost_der = lambda u: u - self.y.reshape(-1, 1)
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
        """Sets the inputs X (batch x p)"""
        self.X = X

        if not init:
            self.features = X.shape[1]
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
        der2 = self.cost_der(self.output[0])
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

        for i in range(self.n_hlayers + 1):
            # add the regularization term to the weights:
            self.weights[i] = self.weights[i] -\
                self.hyperp * self.weights[i]/self.features

            # subtract the gradients dC/dW and dC/db
            self.biases[i] = self.biases[i] - self.eta * diff_b[i]
            self.weights[i] = self.weights[i] - self.eta * diff_w[i]

    def output_func(self):
        """Function to produce an output for a given input"""
        iter = self.act_fn[0](self.X @ self.weights[0] + self.biases[0])
        for l in range(1, self.n_hlayers + 1):
            iter = self.act_fn[l](iter @ self.weights[l] + self.biases[l])
        self.output = iter

    def train_neuron(self, X, y, epochs, save_network=True):
        """Function to train the netorks weights and biases. This function runs
        through one 'epoch' and saves the data as a number between 0 and 1."""
        done = 0
        iterations = int(len(y) / self.batchsize)
        for j in range(epochs):
            for i in range(iterations):
                s = i * self.batchsize  # start
                e = s + self.batchsize  # end
                self.set_inputs(X[s:e, :])
                self.set_outputs(y[s:e], activation="sigmoid")
                self.fb_propogation()
                if str(self.output) == "[nan]":
                    print("Something went wrong. Output is 'nan'...")
                    break
            X, y = fns.shuffle_Xy(X,y,j)  #shuffle after each epoch for stochasticity
            if j%(epochs/10)==0:
                print(f"Epochs: {(j/epochs)*100:.2f}%." + \
                    f" Cost = {self.cost_fn(self.output)}")
        print("--------------")
        print("Network Trained.")
        if save_network:
            self.save_data()

    def test_neuron(self, X, y, load_data, cfn=" ", cumulative_gain=True,\
            confusion_matrix=True):
        """Function the test the networks capabilities"""
        self.batchsize=10
        if self.verbose:
            print("-------------")
            print("Testing:")
        if load_data:
            self.load_data(cfn)
        iterations = int(len(y) / self.batchsize)
        ypred = np.zeros(len(y) - len(y)%self.batchsize)   # account for rests
        for i in range(iterations):
            s = i * self.batchsize  # start
            e = s + self.batchsize  # end
            self.set_inputs(X[s:e, :])
            self.set_outputs(y[s:e], activation="sigmoid")
            self.feedforward()  # feed the network forward once using W and b.
            ypred[s:e] += self.output.reshape(-1,)   # save the outputs

        self.declare_results(y[:e], ypred[:e], round=True)
        print(f"Network had an accuracy of {100*self.acc:.2f} %")
        if cumulative_gain:
            self.plot_cumulative_gain_NNW(y[:e], ypred[:e])
        if confusion_matrix:
            self.plot_confusion_matrix_NNW()

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

    def load_data(self, cfn, fn1="weight_data.npy", fn2="bias_data.npy"):
        """Function to load the weights and biases for a network"""
        pwd = "bin/"
        if cfn == " ":  # if no custom filename is provided:
            self.weights = np.load(pwd + fn1, allow_pickle=True)
            self.biases = np.load(pwd + fn2, allow_pickle=True)
        else:
            self.weights = np.load(pwd + cfn + "1.npy", allow_pickle=True)
            self.biases = np.load(pwd + cfn + "2.npy", allow_pickle=True)
        print("Data loaded.")

    def declare_results(self, ytrue, ypred, round=True):
        """Declare the results, assign the accuracy"""
        if round:
            ypred[np.where(ypred>=0.5)] = 1
            ypred[np.where(ypred<0.5)]  = 0
            for i in range(len(ytrue)):
                if i%20==0:
                    print(f"Output:\t{ypred[i]}\tTrue:\t{ytrue[i]}" + (\
                       " Correct!" if ypred[i]==ytrue[i] else " "))
        else:
            for i in range(len(ytrue)):
                print(f"Output:\t{ypred[i]}\tTrue:\t{ytrue[i]}")

        self.acc = fns.assert_binary_accuracy(ytrue, ypred)

    def plot_cumulative_gain_NNW(self, ytrue, ypred):
        fns.produce_cgchart(ytrue, ypred)

    def plot_confusion_matrix_NNW(self, ytrue, ypred):
        fns.produce_confusion_mtx(ytrue, ypred)

if __name__ == "__main__":
    pass
