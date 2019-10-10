import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics

np.random.seed(2019)


def franke(x, y):
    term = 3 / 4 * np.exp(-(9 * x - 2) ** 2 / 4 - (9 * y - 2) ** 2 / 4)
    term += 3 / 4 * np.exp(-(9 * x + 1) ** 2 / 49 - (9 * y + 1) / 10)
    term += 1 / 2 * np.exp(-(9 * x - 7) ** 2 / 4 - (9 * y - 3) ** 2 / 4)
    term -= 1 / 5 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

    return term


L = 41

X, Y = np.meshgrid(np.linspace(0, 1, L), np.linspace(0, 1, L))
Z = franke(X, Y)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
#
# ax.plot_surface(X, Y, Z)
# ax.set_title("Franke's function")
#
# plt.show()

X_d = np.c_[X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]]
y_d = Z.ravel()[:, np.newaxis]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X_d, y_d, test_size=0.2
)


# Implement neural network
reg = sklearn.neural_network.MLPRegressor(
    hidden_layer_sizes = (100, 20),
    learning_rate = 'adaptive',
    learning_rate_init = 0.01,
    max_iter = 1000,
    tol = 1e-7,
    verbose = True
)
reg.fit(X_train, y_train)

# See some statistics
pred = reg.predict(X_test)
print(f"MSE = {sklearn.metrics.mean_squared_error(y_test, pred)}")
print(f"R2 = {reg.score(X_test, y_test)}")

# Plot surface fit
pred = reg.predict(X_d)
Z_pred = pred.reshape(L, L)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.plot_wireframe(X, Y, Z_pred)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, np.abs(Z-Z_pred))
plt.show()
