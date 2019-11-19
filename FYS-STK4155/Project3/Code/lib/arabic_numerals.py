import xgboost
from xgboost import DMatrix as dmx
from xgboost import train as trn
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


# class ArabicNumerals(XGBClassifier):
#
#     def __init__(self,learning_rate=0.5, max_depth=3,colsample_bytree=0.5,\
#         n_estimators=300,frac=None,k_neighbors=None,m_neighbors=None,\
#         out_step=None, objective='binary:logistic', missing=None):
#
#         if k_neighbors:
#             self.balancingStrategy = 'smote'
#             self.k_neighbors = k_neighbors
#             self.m_neighbors = m_neighbors
#             self.out_step = out_step
#         elif frac :
#             self.balancingStrategy = 'normal'
#             self.frac = frac
#         else:
#             self.balancingStrategy = 'false'
#
#
#         super(ArabicNumerals,self).__init__(seed=500,
#                             learning_rate = learning_rate,
#                             max_depth = max_depth,
#                             colsample_bytree = colsample_bytree,
#                             n_estimators = n_estimators)


class ArabicNumerals(XGBClassifier):
    def __init__(self, my_parameter=10, objective='binary:logistic',\
        missing=None, random_state=0, learning_rate=0.5, max_depth=3,\
        colsample_bytree=0.5,n_estimators=300,frac=None,k_neighbors=None,\
        m_neighbors=None,out_step=None, n_jobs=-1):

        self.my_parameter = 10
        super(ArabicNumerals, self).__init__()


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
    a.fit(X, y.reshape(-1,))
