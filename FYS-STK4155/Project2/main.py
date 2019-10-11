import lib.functions as fns
import lib.logistic_regression as lgr
import time


def main():
    sd = int(time.time())
    fn = "defaulted_cc-clients.xls" # data filename
    X, y = fns.read_in_data(fn, shuffle=True, seed = sd)
    sklearn_beta = fns.sklearn_GDRegressor(X, y).coef_
    print("Sklearn prediction vs. true values:")
    print(X @ sklearn_beta)
    print(y)
    print("-------------------")
    solution = lgr.gradient_descent_solver(X, y, random_state_x0=True)
    print("Customized GD vs. true values:")
    print(X @ solution)
    print(y)

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Completed in {time.time() - start:.2f} seconds.")
