import numpy as np

class Neuron:
    def __init__(self, X, Xt, y, yt, hlayers, act_fn, biases):
        self.X = X
        self.Xt = Xt
        self.y = y
        self.yt = yt
        self.hlayers = hlayers
        self.act_fn = act_fn
        self.biases = biases

    def fb_propogation():
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
