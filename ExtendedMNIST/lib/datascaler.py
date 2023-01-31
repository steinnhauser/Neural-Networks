import pandas as pd
import numpy as np
import sklearn.model_selection as sklms
import sklearn.preprocessing as sklpp


def DataScaler(df, **kwargs):
    """
    Takes in a dataframe and indices which should be
    either scaled or treated categorically. The function
    then scales the training data by itself and the testing
    data by the training data. The function does not
    keep the data sorted. It returns an array where the
    one-hot encoded/categorical columns come first and then
    the scaled ones. The categorical values do not need to
    be scaled, but do not need to be encoded either if they're
    binary. This is what the one-hot index list is for.

    Parameters:
    -----------
    df : DataFrame object
        All the data in one DataFrame
    scol : list
        Scale column indices.
    ncol : list
        Normalize column indices.
    bcol : list
        Binary column indices.
    hcol : list
        One-hot encoded column indices.
    frac : decimal, Default 0.33
        Fraction of testing data. Should be in [0,1]

    Returns:
    --------
    traindf : DataFrame object
        Train data. X and y concatenated
    testdf : DataFrame object
        Test data. X and y concatenated
    """

    scl = kwargs.get("scol", [])
    ncl = kwargs.get("ncol", [])
    hcl = kwargs.get("hcol", [])
    bcl = kwargs.get("bcol", [])
    frc = kwargs.get("frac", 0.50)

    """Raise syntax error messages"""
    msg = "Sum of columns does not match with DataFrame dimensions."
    assert len(hcl+scl+bcl+ncl) == df.shape[1], msg

    msg = "Some values are not accounted for. "\
        + "Should range in total from 0,1,...,p-1"
    assert list(range(df.shape[1])) == sorted(hcl+scl+bcl+ncl), msg

    """Save the column names"""
    col_scale = [df.columns[c] for c in scl]
    col_norm = [df.columns[n] for n in ncl]
    col_bin = [df.columns[b] for b in bcl]
    col_hot = [df.columns[h] for h in hcl]

    """Perform the data splitting before the scaling"""
    train_df, test_df = sklms.train_test_split(df, test_size=frc,\
        shuffle=True)

    total_train = pd.DataFrame()
    total_test  = pd.DataFrame()

    if col_scale:
        """Scale the non-categorical variables:"""
        scaler = sklpp.StandardScaler().fit(train_df[col_scale])
        scaled_train = pd.DataFrame(scaler.transform(train_df[col_scale]),\
            columns=col_scale)
        scaled_test = pd.DataFrame(scaler.transform(test_df[col_scale]), \
            columns=col_scale)

        """Add to the total dataframes"""
        total_train = pd.concat((total_train, scaled_train), axis=1)
        total_test = pd.concat((total_test, scaled_test), axis=1)

    if col_norm:
        """Normalize the non-categorical variables:"""
        normalizer = sklpp.Normalizer().fit(train_df[col_norm])
        normed_train = pd.DataFrame(normalizer.transform(train_df[col_norm]),\
            columns=col_norm)
        normed_test = pd.DataFrame(normalizer.transform(test_df[col_norm]), \
            columns=col_norm)

        """Add to the total dataframes"""
        total_train = pd.concat((total_train, normed_train), axis=1)
        total_test = pd.concat((total_test, normed_test), axis=1)

    """Do nothing with the categorically binary variables:"""
    binary_train = train_df[col_bin]
    binary_test = test_df[col_bin]

    """Combine the data types"""
    total_train = pd.concat((total_train, binary_train.reset_index(drop=True)),\
        axis=1)
    total_test = pd.concat((total_test, binary_test.reset_index(drop=True)),\
        axis=1)

    """Now encode the non-binary categorical data and append it to the DF"""
    for i in col_hot:
        train_dums = pd.get_dummies(train_df[i], prefix=i)
        test_dums = pd.get_dummies(test_df[i], prefix=i)

        """Have to make sure that the features are the same in both the training
        and testing data. Need to insert zeros columns for some cases:"""
        while train_dums.shape[1] != test_dums.shape[1]:
            """If the two don't match, find out which one's lacking, and add
            a column of zeros to that category."""
            if train_dums.shape[1] > test_dums.shape[1]:
                """Use the set '-' operator to find missing elements"""
                missing = set(train_dums.columns) - set(test_dums.columns)
                """Account for there possibly being multiple columns missing"""
                numbers = []
                for j in list(missing):
                    numbers.append(j[-1])

                """Cycle through the columns missing and add zeros"""
                numbers = [int(x) for x in numbers]
                numbers.sort()
                for number in numbers:
                    test_dums.insert(number, i+"_"+str(number), \
                        np.zeros(test_dums.shape[0], dtype=int))

            elif train_dums.shape[1] <= test_dums.shape[1]:
                """Use the set '-' operator to find missing elements. Identical
                to the previous string, only this time the training and test
                dataframes are flipped."""
                missing = set(test_dums.columns) - set(train_dums.columns)
                """Account for there possibly being multiple columns missing"""
                numbers = []
                for j in list(missing):
                    numbers.append(j[-1])

                """Cycle through the columns missing and add zeros"""
                numbers = [int(x) for x in numbers]
                numbers.sort()
                for number in numbers:
                    train_dums.insert(number, i+"_"+str(number), \
                        np.zeros(train_dums.shape[0], dtype=int))

        """Append these to the main DataFrame and proceed to the next"""
        total_train = pd.concat((total_train, train_dums.reset_index(drop=True)), \
            axis=1)
        total_test = pd.concat((total_test, test_dums.reset_index(drop=True)), \
            axis=1)

    return total_train, total_test

if __name__ == '__main__':
    """Testing dataset:"""
    df = pd.DataFrame(np.array([[0, 2, 1, 2, 2, 1],
                                [1, 0, 0, 0, 3, 2],
                                [2, 1, 2, 0, 0, 1],
                                [3, 0, 1, 2, 2, 0],
                                [1, 2, 1, 1, 3, 1],
                                [0, 0, 2, 2, 2, 1],
                                [1, 1, 0, 0, 3, 2],
                                [2, 0, 2, 1, 0, 0],
                                [1, 1, 2, 1, 3, 2],
                                [0, 2, 0, 2, 0, 2],
                                [3, 0, 2, 1, 1, 0],
                                [0, 2, 0, 0, 0, 1]]),\
        columns=["Man", "Woman", "Child", "Old", "Teen", "Mean"])

    # Test the one hot encoding:
    for i in range(1000):
        lists = [[], [], [], []]
        for j in range(6):
            """Randomly declare which of the features is where."""
            dice = np.random.randint(4)
            lists[dice].append(j)
        train, test = DataScaler(df, frac=0.2,\
            hcol=lists[0], ncol=lists[1], bcol=lists[2], scol=lists[3])

        if train.shape[1]==test.shape[1]:
            if i%10==0:
                print(f"\r{i/10+1}%", end="")
        elif train.columns!=test.columns:
            raise ValueError("Train and test columns do not equal.")
        else:
            print(train)
            print(test)
            raise ValueError("A feature was removed.")
    print("")
    print("Test successful.")
    """Works pretty properly. The only disfunctionality is that categories
    which are labeled with 1, 2, 3 etc. are required to start with 0."""

"""
steinn@SHM-PC:~/Desktop/Neural-Networks/FYS-STK4155/Project3/Code/lib$ python3 datascaler.py
100.0%
Test successful.
"""
