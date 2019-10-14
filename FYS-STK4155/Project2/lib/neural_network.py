import numpy as np
import sklearn
import time
import lib.functions as fns

class Neuron:
    def __init__(self, X, y, eta=1, n_hlayers=2, nodes=[576000,100,30,24000],\
            act_str = "sigmoid", biases_str = "zeros", weights_str="zeros",\
            cost_fn_str = "MSE", train_test_split=True, tol_b=20, tol_w=100,\
            maxiter=50):
        if train_test_split:
            self.X, self.Xt, self.y, self.yt=\
            sklearn.model_selection.train_test_split(X, y, \
                    test_size=0.2, random_state=int(time.time()))
        else:
            self.X = X
            self.y = y

        self.eta = eta              # learning rate
        self.n_hlayers = n_hlayers  # no. of hidden layers
        self.nodes = nodes          # list of nodes for each layer
        self.act_str = act_str
        self.set_activation()
        self.biases_str = biases_str
        self.set_biases()
        self.weights_str = weights_str
        self.set_weights()
        self.cost_fn_str = cost_fn_str
        self.set_cost_fn()

        self.tol_b = tol_b  # bias tolerance
        self.tol_w = tol_w  # weight tolerance
        self.maxiter = maxiter  # max number of f/b propogation iterations

    def set_activation(self):
        """Set the activation function of the network."""
        if self.act_str == "sigmoid":
            self.act_fn = np.vectorize(lambda z: 1./(1+np.exp(-z)))
            self.act_der = np.vectorize(lambda z: z*(1-z)) # it's derivative
        elif self.act_str == "tanh":
            self.act_fn = np.vectorize(lambda z: np.tanh(z))
            self.act_der = np.vectorize(lambda z: 1 - np.tanh(z)**2)
        else:
            raise SyntaxError("Activation function must be 'sigmoid' or 'tanh'")

    def set_biases(self):
        """Set the biases of the network."""
        if self.biases_str == "zeros":
            self.biases = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(self.n_hlayers+1):
                # set a bias from each neuron in layer i to each in layer i+1
                self.biases[i] = np.zeros(self.nodes[i+1])
        elif self.biases_str == "random":
            self.biases = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(0, self.n_hlayers+1):
                # set a bias from each neuron in layer i to each in layer i+1
                self.biases[i] = np.random.rand(self.nodes[i+1])
        else:
            raise SyntaxError("Biases must be 'zeros' or 'random'.")

    def set_weights(self):
        """Set the weights of the network."""
        if self.weights_str=="zeros":
            self.weights = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(self.n_hlayers+1):
                # set a weight from each neuron in layer i to each in layer i+1
                self.weights[i] = np.zeros((self.nodes[i+1], self.nodes[i]))
        elif self.weights_str == "random":
            self.weights = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(self.n_hlayers+1):
                # set a weight from each neuron in layer i to each in layer i+1
                self.weights[i] = np.random.rand(self.nodes[i+1], self.nodes[i])
                # this matrix should have dimensions (next layers x prev layers)
        else:
            raise SyntaxError("Weights must be 'zeros' or 'random'.")

    def set_cost_fn(self):
        """Set the cost function"""
        if self.cost_fn_str == "MSE":
            self.cost_fn = lambda u: np.sum(self.y - u)**2
            self.cost_der = lambda u: \
                np.vectorize(-2*(self.y - u)) # derivative of MSE function.
        else:
            raise SyntaxError("Cost function must be 'MSE'.")

    def fb_propogation(self):
        """
        Main program which conducts the forward and backwards
        propogation through the neural network.
        """

        self.err_b = self.tol_b + 1 # initialize these for the first loop
        self.err_w = self.tol_w + 1

        i=0
        print("Step:\t Weight error:\t bias error:\t MSE:")
        while i<self.maxiter and \
            (abs(self.err_b) > self.tol_b or abs(self.err_w) > self.tol_w):
            self.feedforward()
            self.backpropogate()
            print(f"{i}\t{self.err_w:.2f}\t{self.err_b:.2f}\t{self.MSE:.2f}")
            self.produce_outputs()
            i+=1

        if i>=self.maxiter:
            print("Max iteration reached.")
        elif abs(self.err_b) <= self.tol_b and abs(self.err_w) <= self.tol_w:
            print("Bias and weight errors reached tolerance")
        print("---------------------------------------")
        print("Forward/Backward propogation completed.")


    def feedforward(self):
        # Initialize the input 'z' matrix:
        self.z = np.zeros(self.n_hlayers+2, dtype=np.ndarray)
        # Initialize the output 'a' matrix:
        self.a = np.zeros(self.n_hlayers+2, dtype=np.ndarray)

        # Flatten the input data X:
        self.z[0] = self.X.reshape(-1,)
        self.a[0] = self.act_fn(self.z[0])

        for l in range(1, self.n_hlayers+2):
            self.z[l] = self.weights[l-1] @ self.a[l-1] + self.biases[l-1]
            self.a[l] = self.act_fn(self.z[l])

        self.output = self.a[-1]
        self.MSE = np.sum(self.cost_fn(self.output))

    def backpropogate(self):
        # Initialize error vector (one for each layer except input)
        d = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
        for i in range(self.n_hlayers+1):
            d[i] = np.zeros((self.nodes[i+1], self.nodes[i]))

        # Last diff element l=L
        d[-1] = self.act_der(self.output)*2*(self.output- self.y)

        # set up arrays for the differences of bias and weights:
        diff_b = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
        diff_w = np.zeros(self.n_hlayers+1, dtype=np.ndarray)

        diff_b[-1] = d[-1]
        diff_w[-1] = d[-1].reshape(-1,1) @ self.a[-2].reshape(1,-1)

        for l in range(self.n_hlayers-1, -1, -1):
            d[l] = self.weights[l+1].T @ d[l+1]
            diff_b[l] = d[l]
            diff_w[l] = \
                d[l].reshape(-1,1) @ self.act_der(self.z[l]).reshape(1,-1)


        self.err_b = 0
        self.err_w = 0
        for i in range(3):
            self.biases[i] -= self.eta*diff_b[i]
            self.weights[i] -= self.eta*diff_w[i]

            self.err_b += np.sum(diff_b[i])
            self.err_w += np.sum(diff_w[i])

        # The parameter eta is the learning parameter discussed in connection
        # with the gradient descent methods. Here it is convenient to use
        # stochastic gradient descent with mini-batches with an outer loop
        # that steps through the multiple epochs of training. Try to parallize?

    def save_data(self):
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
        pwd = "bin/"
        self.weights = np.load(pwd + fn1, allow_pickle=True)
        self.biases = np.load(pwd + fn2, allow_pickle=True)
        print("Data loaded.")

    def produce_outputs(self, simulate=False):
        if simulate:
            self.feedforward()
        self.acc = fns.assert_binary_accuracy(self.y, self.output)
        print(f"Network produced an accuracy of {100*self.acc:.2f}%")

if __name__ == '__main__':
    pass

"""
steinn@SHM-PC:~/Desktop/Neural-Networks/FYS-STK4155/Project2$ python3 -W ignore main.py
Step:	 Weight error:	 bias error:	 MSE:
0	  50137.50	  3342.50	 44689225.00
.
.
.
98	  45.21	  -10.92	 10150.25
99	  44.90	  -10.85	 10080.09
Max iteration reached.
---------------------------------------
Forward/Backward propogation completed.
"""
