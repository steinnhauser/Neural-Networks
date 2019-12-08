import pandas as pd
import numpy as np
from lib.datascaler import DataScaler
import seaborn as sb
import matplotlib.pyplot as plt
import lib.datasearch as ds
import argparse
from mnist import MNIST

def main():
    """Parse the arguments"""
    argp = argparser()
    p = vars(argp)

    """Possibility to use the mnist python package"""
    # mndata = MNIST('./MNIST_data/original')
    # images, labels = mndata.load_training()

    """Declare the data for analysis"""
    # ds.tiny_eeg_data()
    # ds.MNIST_data(p)
    # ds.EMNIST_data_letters(p)
    ds.EMNIST_data_balanced(p)
    return 1

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cm", "--create_model",
                        help="Used to create a new model.",
                        action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
    """
    Test the functions using:

    $ python3 -m pytest

    """
