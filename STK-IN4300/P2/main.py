import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
from lib.Ex1 import Exercise1
from lib.Ex2 import Exercise2


def main():
    Task1()   # 
    # Task2()     # Done. Bagging with probabilities was not accomplished however.
    return 0

def Task1():
    e1 = Exercise1()
    e1.plot = False     # whether to see the plots or not. There are many.
    print("==============")
    print("Exercise 1.1:")
    print("==============")
    e1.scale_data()             # Exercise 1.   Done.
    print("==============")
    print("Exercise 1.2:")
    print("==============")
    e1.linear_Gauss()           # Exercise 2.   Done.
    print("==============")
    print("Exercise 1.3:")
    print("==============")
    e1.bf_selection1()          # Exercise 3.   Done.
    print("==============")
    print("Exercise 1.4:")
    print("==============")
    e1.boot_CV_compare()        # Exercise 4.   Done.
    print("==============")
    print("Exercise 1.5:")
    print("==============")
    e1.GAM1()                   # Exercise 5.   Done.
    print("==============")
    print("Exercise 1.6:")
    print("==============")
    e1.comp_boosting()          # Exercise 6.   Done. Could not Spline Boost.
    print("==============")
    print("Exercise 1.7:")
    print("==============")
    e1.report_points()          # Exercise 7.   Done.
    print("==============")
    print("Problem 1 Complete.")
    print("==============")

def Task2():
    e2 = Exercise2()
    e2.plot = False     # whether to see plots.
    e2.process_data()           # Necessary for Exercises 1, 2, 3, and 4
    print("==============")
    print("Exercise 2.1:")
    print("==============")
    e2.k_NN()                   # Exercise 1.   Done
    print("==============")
    print("Exercise 2.2:")
    print("==============")
    e2.GAM2()                   # Exercise 2.   Done.
    print("==============")
    print("Exercise 2.3:")
    print("==============")
    e2.tree_bag_ada()           # Exercise 3.   Done.
    print("==============")
    print("Exercise 2.4:")
    print("==============")
    e2.best_method()            # Exercise 4.   Not Done.
    print("==============")
    print("Exercise 2.5:")
    print("==============")
    e2.remove_outliers()        # Exercise 5.   Done.
    print("==============")
    print("Problem 2 Complete.")
    print("==============")

if __name__ == '__main__':
    main()
