import lib.functions as fns
import random
import time


def main():
    sd = int(time.time())
    fn = "defaulted_cc-clients.xls" # insert filename
    X, y = fns.read_in_data(fn, suffle=True, seed = sd)

if __name__ == '__main__':
    start = int(time.time())
    main()
    end = int(time.time())
    t = end - start
    print(f"Completed in {t:.0f} seconds.")
