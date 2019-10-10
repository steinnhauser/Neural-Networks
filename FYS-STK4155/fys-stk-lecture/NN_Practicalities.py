sklearn.neural_network.MLPRegressor(
# Regressor and classifier are the two neural networks from sklearn.
# Fully connected.
# You can specify different activation function for each layer.
# Adam is a stochastic solver with back-propogation.
    hidden_layer_size=(100, 20), # 100 neurons in first layer, 20 in the next.
    learning_rate = 'adaptive', # to lower the learning rate until some threshhold.
    learning_rate_init = 0.01,
    max_iter = 1000, # likely to stop us before this, based on:
    tol = 1e-7, # stop at this.
    verbose = True
)

# TENSOR FLOW
# Usually working with

# KERAS
# Interface which lets you build the NNs in a straight forward manner,
# and choose TF, sklearn etc.
# import TF and use all the conveniances in keras.

# PDES: -g''(x) = f(x)
# boundaries of zero dictate a guess of:
# g(x) = x(1-x)NN(x)
# Try to minimize -g'' - f = 0
# Define a cost function: e.g. MSE.
# MSE = |-g'' - f|**2
