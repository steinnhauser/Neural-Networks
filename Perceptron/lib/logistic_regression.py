import numpy as np
import sklearn


class StochasticGradientDescent:
    """
    Parameters:
    -----------
        beta0 : np.ndarray
            Initial guess for the parameter vector
        gamma : float
            Learning rate of the iterative algorithm in fit(X, y)
        max_iter : int
            Maximum amount of iterations before exiting
        tol : float
            Tolerance which dictates when an answer is sufficient
        verbose : bool, default False
            If True, messages are printed during iterations

    Functions:
    ----------
        fit
        _update_p
        predict
        _generate_random_initial_beta
    """
    def __init__(self, gamma, max_iter, batch_size, verbose=False):
        self.gamma = gamma
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        if self.gamma <= 0:
            raise ValueError("Bad usage:\n\tThe learning rate is negative.")

    def fit(self, X, y, random_initial_beta=True, beta0=None):
        """
        Info:
        -----
        Stochastic gradient descent with mini-batches. Update beta according to:

            beta_new = beta_old - gamma X^T (y - p)

        but for each epoch, loop over number of mini-batches and only
        use the data from that mini-batch.

        Parameters:
        -----------
        X : np.ndarray
            Design matrix
        y : np.ndarray
            Output values
        random_initial_beta : bool
            If True, generate a random initial beta0, if False the initial
            guess should be given as the beta0 parameter
        beta0 : np.ndarray
            Will only be considered if random_initial_beta=False. Provide
            an initial guess for beta0

        Returns:
        --------
        beta : np.ndarray
            Optimal parameter vector
        """
        self.X = X
        self.y = y
        n =  X.shape[0]
        num_batches = int(n/self.batch_size)
        N = num_batches*self.batch_size # ignore the last (n%batch_size) data points
        batches = np.random.permutation(N).reshape(num_batches, self.batch_size)

        if random_initial_beta:
            self._generate_random_initial_beta()
        else:
            if not beta0:
                raise ValueError("Bad usage:\n\tbeta0 must be provided" + \
                " when random_initial_beta=False.")
            else:
                self.beta = beta0

        if self.beta.shape[0] != self.X.shape[1]:
            raise ValueError(f"Mismatch: beta has {self.beta.shape[0]}" + \
            f"  features, X has {self.X.shape[1]} columns")

        t0 = 1
        t1 = 10
        learning_rate = lambda t: t0/(t + t1)
        self.bias = 0.1

        for iter in range(self.max_iter): # epochs
            for j in range(num_batches): # loop over batches
                indices = batches[np.random.randint(num_batches),:] # random batch
                _p = self._update_p(self.X[indices,:]) #
                y_p = (self.y[indices] - _p)
                self.bias -= np.mean(y_p)
                gamma = learning_rate(iter*num_batches + j)
                step = gamma * self.X[indices,:].T @ y_p
                self.beta -= step

            norm = np.linalg.norm(step)
            if self.verbose:
                print(f"Step {i:5}: norm(beta_j) = {norm:1.4e}")

        print(f"SGD done:\nnorm(beta) = {norm:1.4e}")

        return None

    def _update_p(self, Xi):
        """
        Updates the probability vector using the sigmoid function.
        """
        return 1.0 / (1 + np.exp(- Xi @ self.beta + self.bias))

    def predict(self, Xt):
        """
        Info:
        -----
        Predict outputs yp given input values in Xt.
        Note: fit must have been called before predict can be done!

        Returns:
        --------
        yp : np.ndarray
            Predicted outputs
        """
        yp = self._update_p(Xt)
        return yp

    def _generate_random_initial_beta(self):
        """
        Info:
        -----
        Generate a random initial parameter vector

        Returns:
        --------
            Updates self.beta0
        """
        predictors = self.X.shape[1]  # p
        self.beta = (np.random.random(predictors) - 0.5) * 1.4 #  in [-0.7, 0.7]
        return None

if __name__ == '__main__':
    pass
