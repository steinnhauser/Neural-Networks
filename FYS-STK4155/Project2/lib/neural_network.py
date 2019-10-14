import numpy as np

class Neuron:
    def __init__(self, X, Xt, y, yt, n_hlayers=2, nodes=[24,100,30,2], \
            act_str = "sigmoid", biases_str = "random", weights_str="random"):
        self.X = X      # testing data
        self.Xt = Xt    # training data
        self.y = y
        self.yt = yt
        self.n_hlayers = n_hlayers      # no. of hidden layers
        self.nodes = nodes              # list of nodes for each layer
        self.act_str = act_str
        self.set_activation()
        self.biases_str = biases_str
        self.set_biases()
        self.weights_str = weights_str
        self.set_weights()


    def set_activation(self):
        """Set the activation function of the network."""
        if self.act_str == "sigmoid":
            self.act_fn = np.vectorize(lambda z: 1./(1+np.exp(-z)))
        elif self.act_str == "tanh":
            self.act_fn = np.vectorize(lambda z: np.tanh(z))
        else:
            raise SyntaxError("Activation function must be 'sigmoid' or 'tanh'")

    def set_biases(self):
        """Set the biases of the network."""
        if self.biases_str == "zeros":
            self.biases = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(self.n_hlayers+1):
                self.biases[i] = np.zeros(self.nodes[i+1])
        elif self.biases_str == "random":
            self.biases = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(self.n_hlayers+1):
                self.biases[i] = np.random.rand(self.nodes[i+1])
        else:
            raise SyntaxError("Biases must be 'zeros' or 'random'.")

    def set_weights(self):
        """Set the weights of the network."""
        if self.weights_str=="zeros":
            self.weights = np.zeros(n_hlayers+1, dtype=np.ndarray)
            for i in range(self.n_hlayers+1):
                self.weights[i] = np.zeros(self.nodes[i+1])
        elif self.weights_str == "random":
            self.weights = np.zeros(self.n_hlayers+1, dtype=np.ndarray)
            for i in range(self.n_hlayers+1):
                self.weights[i] = np.random.rand(self.nodes[i+1])
        else:
            raise SyntaxError("Weights must be 'zeros' or 'random'.")

    def fb_propogation(self):
        """
        Main program which conducts the forward and backwards
        propogation through the neural network.
        """
        if random_weights:
            for i in range(self.n_hlayers+1):
                pass
                # Initialize weights for n+1 hidden layers.
                # Generate a (n_nodes x 1) weights array for the layers.



        # Initialize:
        #   Input data X and activations z[1] of the input layer and compute
        #   the activation function and the pertinent outputs a[1]

        # Secondly:
        #   We perform the feed torward till we reach the output layer and
        #   compute all z[l] of the input layer and compute the activation
        #   function and the pertinent outputs a[l] for l=2, 3, ..., L.

        # Thereafter:
        #   We compute the output error d[l] by computing all:
        #   d[L,j] = f'(z[L,j])*(dC/d(a[L,j]))

        # Subsequently:
        #   Compute the back propogation error for each l= L-1, L-2, ..., 2
        #   d[l, j] = 0
        #   for i in range(K):
        #       d[l,j] += d[l+1, k] * w[l+1, k, j] * f'(z[l,j])

        # Finally:
        #   We update the weights and the biases using gradient descent
        #   for each l=L-1, L-2, ..., 2 and update the weights and
        #   biases according to the rules:
        #       w[l,j,k] <- w[l,j,k] - eta*d[l,j]*a[l-1,k]
        #       b[l,j] <- b[l,j]-eta*(dC/d(b[l,j])) = b[l,j]-eta*d[l,j]

        # The parameter eta is the learning parameter discussed in connection
        # with the gradient descent methods. Here it is convenient to use
        # stochastic gradient descent with mini-batches with an outer loop
        # that steps through the multiple epochs of training. Try to parallize?

if __name__ == '__main__':

    neuron = Neuron()
