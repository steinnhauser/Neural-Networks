import xgboost
from xgboost import DMatrix as dmx
from xgboost import train as trn
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sb
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


class ArabicNumerals(XGBClassifier):
    def __init__(self, objective='reg:logistic', missing=None, \
        random_state=0, learning_rate=0.5, max_depth=3, colsample_bytree=0.5,\
        n_estimators=300, frac=None, k_neighbors=None, m_neighbors=None,\
        out_step=None, n_jobs=-1, modelname="MNIST_model"):

        self.modelname = modelname

        if k_neighbors:
            self.balancingStrategy = 'smote'
            self.k_neighbors = k_neighbors
            self.m_neighbors = m_neighbors
            self.out_step = out_step
        elif frac:
            self.balancingStrategy = 'normal'
            self.frac = frac
        else:
            self.balancingStrategy = 'false'

        super(ArabicNumerals, self).__init__(
            seed=500,
            learning_rate = learning_rate,
            max_depth = max_depth,
            colsample_bytree = colsample_bytree,
            n_estimators = n_estimators
        )

    def set_data(self, Xtrain, ytrain, Xtest, ytest):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest

    def fit_data(self):
        self.fit(self.Xtrain, self.ytrain)  # .reshape(-1,)

    def predict_data(self):
        self._le = LabelEncoder().fit(self.ytest)
        self.ypred = self.predict(self.Xtest)

    def save(self):
        self.save_model("MNISTmodels/" + self.modelname)

    def load(self):
        self.load_model("MNISTmodels/" + self.modelname)

    def assert_performance(self):
        """Function to assert the performance of the prediction
        ypred in relation to the testing data ytest."""
        from sklearn.metrics import accuracy_score, jaccard_score, zero_one_loss

        self.accuracy_score = accuracy_score(self.ytest, self.ypred)
        self.Jaccard_index  = jaccard_score(self.ytest, self.ypred,\
            average = None)
        self.zero_one_loss  = zero_one_loss(self.ytest, self.ypred)

        print(f"Performance of {self.modelname} summary:")
        print(f"\tAccuracy score: {self.accuracy_score*100:.2f}%")
        print(f"\tJaccard index average: {np.mean(self.Jaccard_index):.2f}")
        print(f"\tZero one loss: {self.zero_one_loss:.2f}")

    def incorrect_labels(self):
        """Function which returns the labels of the data
        which was incorrectly categorized."""
        miscat_inds = np.where(list(self.ypred)!=self.ytest.values.reshape(-1,))
        miscat_vals = self.ypred[miscat_inds[0]]
        self.miscat_vals = miscat_vals
        self.miscat_inds = miscat_inds[0]

    def illustrate_performance(self):
        """Plots the confusion matrix"""
        from sklearn.metrics import confusion_matrix
        cmtx = confusion_matrix(self.ytest, self.ypred)
        ax = sb.heatmap(data = cmtx, annot=True, fmt=".0f")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)
        plt.title(f"Confusion Matrix of model {self.modelname}.")
        plt.ylabel("Predicted data")
        plt.xlabel("Testing data")
        plt.show()

    def illustrate_label(self, ind):
        """Input one 28x28 series for grayscale plotting"""
        index = ind
        data = self.Xtest.iloc[ind].values.reshape(28,28)
        ax = sb.heatmap(data, cmap="gray", cbar=False)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        plt.title(  f"Image of index no. {index}."+"\n"\
                    f"A {self.ytest.iloc[ind].values[0]} misclassified" +\
                    f" as a {self.ypred[ind]}")
        plt.show()

if __name__ == '__main__':
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

    X = df.loc[:,df.columns!="Mean"].values
    y = df.loc[:,df.columns=="Mean"].values
    a = ArabicNumerals()
    # a.fit(X, y.reshape(-1,))
    a.set_data(X, y)
    a.fit_data()