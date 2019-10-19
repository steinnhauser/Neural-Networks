import numpy as np
import sklearn
import time
import lib.functions as fns

class Neuron:
    def __init__(self, eta=0.1, n_hlayers=3, nodes=[24, 18, 12, 6, 1],\
            act_str = "relu", biases_str = "random", weights_str="random",\
            cost_fn_str = "MSE", tol_bw=1e-5, datasets_sampled=0, verbose=False,\
            tol_mse=1e-4, maxiter=20):

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

        self.datasets_sampled = datasets_sampled # sets the network has been trained on
        self.verbose = verbose      # print unnessecary indications
        self.tol_bw = tol_bw        # bias and weight diff tolerance
        self.tol_mse = tol_mse      # mse tolerance
        self.maxiter = maxiter      # max number of f/b propogation iterations

    def set_activation(self):
        """Set the activation function of the network."""
        if self.act_str == "sigmoid":
            self.act_fn = np.vectorize(lambda z: 1./(1+np.exp(-z)))
            self.act_der = \
                np.vectorize(lambda z: self.act_fn(z)*(1-self.act_fn(z)))
        elif self.act_str == "tanh":
            self.act_fn = np.vectorize(lambda z: np.tanh(z))
            self.act_der = np.vectorize(lambda z: 1 - np.tanh(z)**2)
        elif self.act_str == "relu":
            self.act_fn = np.vectorize(lambda z: 0 if z<0. else z)
            self.act_der = np.vectorize(lambda z: 0 if z<0. else 1)
        else:
            raise \
            SyntaxError("Activation function must be 'sigmoid'\
                or 'tanh' or 'relu'")

    def set_biases(self):
        """Set the biases of the network."""
        if self.biases_str == "zeros":
            self.biases = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(self.n_hlayers+1):
                self.biases[i] = np.zeros(self.nodes[i+1])
        elif self.biases_str == "random":
            self.biases = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(0, self.n_hlayers+1):
                self.biases[i] = np.random.rand(self.nodes[i+1]) - 0.5
        elif self.biases_str == "custom":
            pass
        else:
            raise SyntaxError("Biases must be 'zeros' or 'random'.")

    def set_weights(self):
        """Set the weights of the network."""
        if self.weights_str=="zeros":
            self.weights = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(self.n_hlayers+1):
                self.weights[i] = np.zeros((self.nodes[i+1], self.nodes[i]))
        elif self.weights_str == "random":
            self.weights = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(self.n_hlayers+1):
                self.weights[i] = np.random.rand(self.nodes[i+1],self.nodes[i])-0.5
                # this matrix should have dimensions (next layers x prev layers)
        elif self.weights_str == "custom":
            pass
        else:
            raise SyntaxError("Weights must be 'zeros' or 'random'.")

    def set_cost_fn(self):
        """Set the cost function"""
        if self.cost_fn_str == "MSE":
            self.cost_fn = lambda u: 0.5*(self.y - u)**2
            self.cost_der = lambda u: np.vectorize(-(self.y - u))
        elif self.cost_fn_str == "xentropy":
            self.cost_fn = lambda u: -self.y*np.log(u)
            self.cost_der = lambda u: u - self.y
        elif self.cost_fn_str == "softmax":
            self.cost_fn = lambda u: np.exp(u) / np.sum(np.exp(u))
            self.cost_der = lambda u: self.cost_fn(u)*(1-self.cost_fn(u))
        else:
            raise SyntaxError("Cost function must be 'MSE' or 'xentropy'\
                or 'softmax'.")

    def set_inputs_outputs(self, X, y):
        """Sets the inputs X and outputs y. Arranges nodes respectively."""
        self.X = X
        self.y = y
        if len(self.nodes) == self.n_hlayers:
            self.nodes = [len(X)] + self.nodes + [len(y)] # add input/output
        elif len(self.nodes) == self.n_hlayers+2:
            pass

    def fb_propogation(self):
        """
        Main program which conducts the forward and backwards
        propogation through the neural network for training.
        """

        self.err_bw = self.tol_bw + 1 # initialize this for the first loop
        self.datasets_sampled+=1

        i=0
        while i<self.maxiter and self.err_bw > self.tol_bw:
            self.feedforward()
            self.backpropogate()
            i+=1

        if self.verbose:
            print(f"Output:\t{self.output_func()}\tTrue:\t{self.y}")
            if self.err_bw <= self.tol_bw:
                print("Weight and bias difference tolerance reached.")
            else:
                print("Max iteration reached.")

    def feedforward(self):
        # Initialize the input 'z' matrix:
        self.z = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
        # Initialize the output 'a' matrix:
        self.a = np.zeros(self.n_hlayers+1, dtype=np.ndarray)

        # initialize first ('zeroth') layer
        self.z[0] = self.weights[0] @ self.X + self.biases[0]
        self.a[0] = self.act_fn(self.z[0])

        for l in range(1, self.n_hlayers+1):
            self.z[l] = self.weights[l] @ self.a[l-1] + self.biases[l]
            self.a[l] = self.act_fn(self.z[l])

        self.output = self.a[-1]

    def backpropogate(self):
        # Initialize error vector delta (one for each layer except input)
        d = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
        for i in range(self.n_hlayers+1):
            # have difference array for each layer:
            d[i] = np.zeros(self.nodes[i+1])

        # set up arrays for the differences of bias and weights:
        diff_b = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
        diff_w = np.zeros(self.n_hlayers+1, dtype=np.ndarray)

        # Last diff element l=L
        der1 = self.act_der(self.z[-1])
        der2 = -2*(self.y - self.output)
        d[-1] = np.multiply(der1, der2) # element wise multiplication.

        # first elements of the difference arrays:
        diff_b[-1] = d[-1]
        diff_w[-1] = np.outer(d[-1], self.a[-2])    # a[] 'previous layer'

        for l in range(self.n_hlayers-1, -1, -1):
            d[l] = np.multiply((self.weights[l+1].T @ d[l+1]),\
                self.act_der(self.z[l]))
            diff_b[l] = d[l]
            diff_w[l] = np.outer(d[l], self.a[l-1])   # a[] 'previous layer'

        self.err_bw = 0
        for i in range(self.n_hlayers+1):
            self.biases[i] -= np.multiply(self.eta, diff_b[i])
            self.weights[i] -= np.multiply(self.eta, diff_w[i])

            self.err_bw += abs(np.sum(diff_b[i])) + abs(np.sum(diff_w[i]))

        # The parameter eta is the learning parameter discussed in connection
        # with the gradient descent methods. Here it is convenient to use
        # stochastic gradient descent with mini-batches with an outer loop
        # that steps through the multiple epochs of training. Try to parallize?

    def output_func(self):
        iter = self.act_fn(self.weights[0] @ self.X + self.biases[0])
        for l in range(1, self.n_hlayers+1):
            iter = self.act_fn(self.weights[l] @ iter + self.biases[l])
        return iter

    def train_neuron(self, X, y, train_no=100, Xscale=True):
        done = 0
        for s in range(train_no):
            if Xscale==True:
                Xd = sklearn.preprocessing.scale(X[s,:])
            else:
                Xd = X[s,:]
            self.set_inputs_outputs(Xd, y[s])
            self.fb_propogation()
            if str(self.output) == '[nan]':
                print("Something went wrong. Output is 'nan'...")
                break
            if self.err_bw < self.tol_bw and done==0:
                print(f"Network is trained up to tolerance after {s} sets.")
                done+=1
        print("--------------")
        print("Network Trained.")
        self.save_data()

    def test_neuron(self, X, y, test_no = 100, load_data=True, Xscale=True):
        if load_data:
            self.load_data()
        correct = 0
        for s in range(test_no):
            if Xscale==True:
                Xd = sklearn.preprocessing.scale(X[s,:])
            else:
                Xd = X[s,:]
            self.set_inputs_outputs(Xd, y[s])
            self.feedforward()  # feed the network forward once using W and b.
            if (self.output>0.5 and y[s]==1) or (self.output<0.5 and y[s]==0):
                correct += 1
        print(f"Network had an accuracy of {correct/test_no*100:.2f} %")


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

    def assert_accuracy(self, X, y, test_sample=100):
        results = np.zeros(test_sample)
        for i in range(test_sample):
            self.X = X[i,:]
            self.feedforward()
            results[i] = self.output

        # results = (np.random.random(test_sample))*0.7/0.5 # test this as results.
        self.acc = fns.assert_binary_accuracy(y[:test_sample], results)
        print(f"Network produced an accuracy of {100*self.acc:.2f}%")

if __name__ == '__main__':
    pass

"""
steinn@SHM-PC:~/Desktop/Neural-Networks/FYS-STK4155/Project2$ python3 -W ignore main.py
--------------
Network Trained.
Network had an accuracy of 79.50 %
-------------------
Completed in 190.83 seconds.

"""
