import xgboost
from xgboost import DMatrix as dmx
from xgboost import train as trn
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sb
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


class ArabicAlphabet(XGBClassifier):
    def __init__(self, objective='reg:logistic', missing=None, \
        random_state=0, learning_rate=0.5, max_depth=3, colsample_bytree=0.5,\
        n_estimators=300, frac=None, k_neighbors=None, m_neighbors=None,\
        out_step=None, n_jobs=-1, modelname="EMNIST_model", verbose=False):

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

        super(ArabicAlphabet, self).__init__(
            seed=500,
            learning_rate = learning_rate,
            max_depth = max_depth,
            colsample_bytree = colsample_bytree,
            n_estimators = n_estimators,
            verbose = verbose
        )

    def set_data(self, Xtrain, ytrain, Xtest, ytest):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest

    def fit_data(self):
        self.fit(self.Xtrain, self.ytrain)  # .reshape(-1,)

    def predict_data(self):
        self._le = LabelEncoder().fit(self.ytrain)
        self.ypred = self.predict(self.Xtest)

    def save(self):
        self.save_model("EMNISTmodels/" + self.modelname)

    def load(self):
        self.load_model("EMNISTmodels/" + self.modelname)

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
        self.miscat_vals = miscat_vals # from hexa to alphabet
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
        data = self.Xtest.iloc[ind].values.reshape(28,28).T
        ax = sb.heatmap(data, cmap="gray", cbar=False)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        plt.title(
            f"Image of index no. {index}."+"\n"
            f"'{self.ytest.iloc[ind]}' was misclassified" +
            f" as a '{self.ypred[ind]}'"
        )
        plt.show()

    def illustrate_label_grid(self):
        """Function which illustrates 9 plots of misclassifications picked
        randomly from the 'miscat_inds' list."""

        # randints = np.random.randint(len(self.miscat_inds), size=9)
        """Need to implement these randints. There are only misclassifications
        of 'a' being illustrated currently."""

        ax = plt.subplot(331)
        sb.heatmap(self.Xtest.iloc[self.miscat_inds[0]].values.reshape(28,28).T,
            cmap="gray", ax=ax, cbar=False)
        plt.title(f"{self.ytest.iloc[self.miscat_inds[0]]} misclassified as"+
            f" {self.ypred[self.miscat_inds[0]]}")
        ax.set_xticks([])
        ax.set_yticks([])
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)

        ax = plt.subplot(332)
        sb.heatmap(self.Xtest.iloc[self.miscat_inds[1]].values.reshape(28,28).T,
            cmap="gray", ax=ax, cbar=False)
        plt.title(f"{self.ytest.iloc[self.miscat_inds[1]]} misclassified as"+
            f" {self.ypred[self.miscat_inds[1]]}")
        ax.set_xticks([])
        ax.set_yticks([])
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)

        ax = plt.subplot(333)
        sb.heatmap(self.Xtest.iloc[self.miscat_inds[2]].values.reshape(28,28).T,
            cmap="gray", ax=ax, cbar=False)
        plt.title(f"{self.ytest.iloc[self.miscat_inds[2]]} misclassified as"+
            f" {self.ypred[self.miscat_inds[2]]}")
        ax.set_xticks([])
        ax.set_yticks([])
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)

        ax = plt.subplot(334)
        sb.heatmap(self.Xtest.iloc[self.miscat_inds[3]].values.reshape(28,28).T,
            cmap="gray", ax=ax, cbar=False)
        plt.title(f"{self.ytest.iloc[self.miscat_inds[3]]} misclassified as"+
            f" {self.ypred[self.miscat_inds[3]]}")
        ax.set_xticks([])
        ax.set_yticks([])
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)

        ax = plt.subplot(335)
        sb.heatmap(self.Xtest.iloc[self.miscat_inds[4]].values.reshape(28,28).T,
            cmap="gray", ax=ax, cbar=False)
        plt.title(f"{self.ytest.iloc[self.miscat_inds[4]]} misclassified as"+
            f" {self.ypred[self.miscat_inds[4]]}")
        ax.set_xticks([])
        ax.set_yticks([])
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)

        ax = plt.subplot(336)
        sb.heatmap(self.Xtest.iloc[self.miscat_inds[5]].values.reshape(28,28).T,
            cmap="gray", ax=ax, cbar=False)
        plt.title(f"{self.ytest.iloc[self.miscat_inds[5]]} misclassified as"+
            f" {self.ypred[self.miscat_inds[5]]}")
        ax.set_xticks([])
        ax.set_yticks([])
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)

        ax = plt.subplot(337)
        sb.heatmap(self.Xtest.iloc[self.miscat_inds[6]].values.reshape(28,28).T,
            cmap="gray", ax=ax, cbar=False)
        plt.title(f"{self.ytest.iloc[self.miscat_inds[6]]} misclassified as"+
            f" {self.ypred[self.miscat_inds[6]]}")
        ax.set_xticks([])
        ax.set_yticks([])
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)

        ax = plt.subplot(338)
        sb.heatmap(self.Xtest.iloc[self.miscat_inds[7]].values.reshape(28,28).T,
            cmap="gray", ax=ax, cbar=False)
        plt.title(f"{self.ytest.iloc[self.miscat_inds[7]]} misclassified as"+
            f" {self.ypred[self.miscat_inds[7]]}")
        ax.set_xticks([])
        ax.set_yticks([])
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)

        ax = plt.subplot(339)
        sb.heatmap(self.Xtest.iloc[self.miscat_inds[8]].values.reshape(28,28).T,
            cmap="gray", ax=ax, cbar=False)
        plt.title(f"{self.ytest.iloc[self.miscat_inds[8]]} misclassified as"+
            f" {self.ypred[self.miscat_inds[8]]}")
        ax.set_xticks([])
        ax.set_yticks([])
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)

        plt.show()


if __name__ == '__main__':
    pass
