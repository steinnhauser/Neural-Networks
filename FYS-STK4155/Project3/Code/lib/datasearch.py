import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from lib.datascaler import DataScaler
from lib.arabic_numerals import ArabicNumerals
from lib.arabic_alphabet import ArabicAlphabet


def tiny_eeg_data():
    """Begin with preprocessing the data"""
    pwd = "tiny_eeg_data/tiny_eeg_self_experiment_reading.csv"
    df = pd.read_csv(pwd)
    """
    Experimental data found on Kaggle:
        https://www.kaggle.com/millerintllc/eeg-microexperiment

    The features are listed as so:
        * 1. IndexId  - Row nr.
    Raw EEG signals:
        * 2. Channel1 - Decimal
        * 3. Channel2 - Decimal
        * 4. Channel3 - Decimal
        * 5. Channel4 - Decimal
    Constants:
        * 6. Ref1     - Integer
        * 7. Ref2     - Integer
        * 8. Ref3     - Integer
    Two timestamps:
        * 9. TS1      - Date
        * 10. TS2     - Decimal
    """
    df = df.loc[:,df.columns!="IndexId"] # remove the index row
    df = df.loc[:,df.columns!="TS2"] # only values of 1.5e12
    df = df.fillna(method='ffill')
    df["TS1"] = df["TS1"].apply(lambda time:\
        sum(x * int(t) for x, t in zip([3600, 60, 1], str(time).split(":"))))

    """Need to normalize the constant values"""
    ncol = [4, 5, 6, 7]
    """Need to scale the decimal values."""
    scol = [0, 1, 2, 3]
    train, test = DataScaler(df, frac=0.2, scol=scol, ncol=ncol)

    Channels = train.loc[:,["Channel1", "Channel2", "Channel3", "Channel4"]]
    ax = sb.heatmap(Channels.corr().round(2), annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)    # Fix edges
    plt.show()

def MNIST_data(p):
    train = pd.read_csv("MNIST_data/csv/mnist_train.csv")
    test = pd.read_csv("MNIST_data/csv/mnist_test.csv")

    """
    Values range from 0 to 255, and the first
    column is the label of the image. Data shapes are:
    (60000, 785)
    (10000, 785)
    """

    X_train = train.loc[:, train.columns!="label"]
    y_train = train.loc[:, train.columns=="label"]
    X_test = test.loc[:, test.columns!="label"]
    y_test = test.loc[:, test.columns=="label"]

    a = ArabicNumerals(modelname = "MNIST_2", verbose=True)

    if p["create_model"]:
        print("Generating model...")
        a.set_data(X_train, y_train, X_test, y_test)
        a.fit_data()
        a.save()
    else:
        print("Loading model...")
        a.set_data(X_train, y_train, X_test, y_test)
        a.load()

    a.predict_data()
    a.assert_performance()
    a.illustrate_performance()
    a.incorrect_labels()
    # To illustrate some of the misclassifications:
    a.illustrate_label(a.miscat_inds[0])

def EMNIST_data(p):
    train = pd.read_csv("EMNIST_data/emnist_csv/emnist-letters-train.csv")
    test = pd.read_csv("EMNIST_data/emnist_csv/emnist-letters-test.csv")
    """
    Values range from 0 to 255, and the first
    column is the label of the image. Data shapes are:
    (88799, 785)
    (14799, 785)
    """
    # set the labels to a more reasonable format
    labels = ["label"]
    for i in range(1, 29):
        for j in range(1, 29):
            labels.append(str(j)+"x"+str(i))

    train.columns = labels
    test.columns = labels
    test = test.sample(frac=1)

    """Strangely enough, the max value of the testing data is only up to 19."""

    X_train = train.loc[:, train.columns!="label"]
    y_train = train.loc[:, train.columns=="label"]
    X_test = test.loc[:, test.columns!="label"]
    y_test = test.loc[:, test.columns=="label"]

    # transform the first row to be alphabetical instead of hexadecimal:
    import string
    ascii_list = list(string.ascii_lowercase)
    y_train = pd.Series(y_train["label"]).apply(lambda x: str(ascii_list[x-1]))
    y_test = pd.Series(y_test["label"]).apply(lambda x: str(ascii_list[x-1]))

    a = ArabicAlphabet(modelname = "EMNIST_2", verbose=True)

    if p["create_model"]:
        print("Generating model...")
        a.set_data(X_train, y_train, X_test, y_test)
        a.fit_data()
        a.save()
    else:
        print("Loading model...")
        a.set_data(X_train, y_train, X_test, y_test)
        a.load()

    a.predict_data()
    a.assert_performance()
    a.illustrate_performance()
    a.incorrect_labels()
    a.illustrate_label(a.miscat_inds[0]) # illustrate some misclassifications

if __name__ == '__main__':
    # tiny_eeg_data()
    # MNIST_data()
    pass
