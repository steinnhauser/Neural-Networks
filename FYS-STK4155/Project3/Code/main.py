import pandas as pd
import numpy as np
from lib.datascaler import DataScaler
import seaborn as sb
import matplotlib.pyplot as plt
import lib.datasearch as ds

def main():
    ds.tiny_eeg_data()
    return 1


if __name__ == "__main__":
    main()
    """
    Test the functions using:

    $ python3 -m pytest

    """
