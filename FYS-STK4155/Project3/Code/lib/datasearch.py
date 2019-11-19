import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from lib.datascaler import DataScaler

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



if __name__ == '__main__':
    tiny_eeg_data()
