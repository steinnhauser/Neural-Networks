import lib.functions as fns
import time


def main():
    sd = int(time.time())
    fn = "defaulted_cc-clients.xls" # data filename
    X, y = fns.read_in_data(fn, shuffle=True, seed = sd)
    sklearn_beta = fns.sklearn_GDRegressor(X, y).coef_
    print(X @ sklearn_beta)
    print(y)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f"Completed in {end - start:.2f} seconds.")
