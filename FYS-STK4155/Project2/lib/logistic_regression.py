import numpy as np
import sklearn


def logistic_regression():
    # need to determine:
    #   Cost function
    #   Design matrix

    # Implement a gradient descent solver. Either:
    #   Standartd GD with learning rate, or
    #   Attempt Newton-Raphson method.

    # May be useful to implement a Stochastic Gradient Descent method which has
    # an argument "mini-batches". This is useful for the Neural Network later.
    pass


class GradientDescent:
    def __init__(
        self,
        x0=0,
        random_state_x0=False,
        gamma_k=0.01,
        max_iter=50,
        tol=1e-5,
        verbose=False,
    ):
        self.x0 = x0
        self.gamma_k = gamma_k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state_x0 = random_state_x0
        self.verbose = verbose

    def solve(self, X, y):
        self.X = X
        self.y = y
        self.gradient_descent_solver()

    def gradient_descent_solver(self):
        """
        Calculates a gradient descent starting from x0.

        Parameters:
        -----------
        x0 : vec
            Initial guess for the minimum of F
        gamma_k : float
            Learning rate of the solver ('step size' of delF).
        max_iter : int
            Maximum amount of iterations before exiting.
        tol : float
            Tolerance which dictates when an answer is sufficient.

        Returns:
        --------
        xsol : vec
            Vector which produces a minimum of F.
        """
        if self.gamma_k <= 0:
            raise ValueError("Bad useage:\n\tThe learning rate is negative.")

        if self.random_state_x0:
            preds = X.shape[1]  # p
            self.xsol = (
                (np.random.random(preds) - 0.5) * 0.7 / 0.5
            )  # between [-0.7, 0.7]
            if not type(self.x0) == int:
                if not np.equal(self.x0.all(), 0):
                    print("Useage Warning: Overwriting set x0 with random values.")
            elif type(self.x0) == int and self.x0 != 0:
                print(f"Useage Warning: Overwriting set x0={x0} with random values")

        else:
            if type(self.x0) == int:
                if self.x0:
                    self.xsol = np.ones(self.X.shape[1]) * self.x0
                else:
                    self.xsol = np.zeros(self.X.shape[1])
            elif type(self.x0) == np.ndarray:
                if self.x0.shape[0] == self.X.shape[1]:  # if len = p
                    self.xsol = self.x0
                else:
                    raise ValueError("Bad useage: x0 was not of length 1 or p.")
            else:
                raise ValueError(
                    "Bad useage: x0 was not of type 'int' or 'numpy.ndarray'"
                )

        # calculate the first step
        self.calculate_p()
        self.delF()
        self.step = self.gamma_k * self.dF

        i = 0
        while i <= self.max_iter:
            self.xsol = self.xsol - self.step
            # calculate the next step
            self.calculate_p()
            self.delF()
            self.step = self.gamma_k * self.dF
            if np.linalg.norm(self.step) <= self.tol:
                print("GD reached tolerance.")
                break
            if self.verbose:
                print(f"{i}\t{np.linalg.norm(self.step)}")
            i += 1

        if i >= self.max_iter:
            print("GD reached max iteration.")

    def delF(self):
        """
        Calculates an estimation of the gradient of the cost function F.

        Parameters:
        -----------
        x : vec
            Input which dictates where gradient should work from

        Returns:
        --------
        dF : vec
            Output of which direction F decreases in.
        """
        a = self.y - self.p
        self.dF = -self.X.T @ a

    def calculate_p(self):
        """
        Calculates the probability vector using the sigmoid function.
        """
        fac = self.X @ self.xsol
        self.p = 1.0 / (1 + np.exp(-fac))  # np.exp(fac)(1+np.exp(fac))is strange

    def predict(self, Xt, yt):
        yp = Xt @ obj.xsol
        a = fns.assert_binary_accuracy(yt, yp)
        print(f"GDS had accuracy of {100*a:.0f} %")


if __name__ == "__main__":
    pass
    # logistic_regression()
    # CGMethods()
