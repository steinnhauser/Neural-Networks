import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import imblearn.over_sampling
from scikitplot.metrics import plot_cumulative_gain


class Exercise2:
    def __init__(self):
        pass

    def process_data(self):
        """
        pregnant    : number of pregnancies;
        glucose     : plasma glucose concentration at 2 h in an oral glucose
        tolerance test;
        pressure    : diastolic blod pressure (mm Hg);
        triceps     : triceps skin fold thickness (mm);
        insulin     : 2-h serum insulin (μU/mL);
        mass        : body mass index (kg/m 2 );
        pedigree    : diabetes pedigree function;
        age         : age (years)
        """
        df = pd.read_csv("diabetes.csv")


        train_df, test_df =\
            sklearn.model_selection.train_test_split(df, test_size=0.33,\
                stratify=df["Outcome"]) # stratify such that the outcome is even

        """Need to check if the number of 1's and 0's is equal. If this is not
        the case, we need to upsample the training data."""

        # yval = train_df["Outcome"].values
        # ones = len(yval[np.where(yval>0.5)])
        # zers = len(yval[np.where(yval<0.5)])
        # print(ones) # 134
        # print(zers) # 250

        """Need to upsample the 'ones' cases for the training data"""
        Xtrain = train_df.loc[:,train_df.columns!="Outcome"]
        ytrain = train_df.loc[:,["Outcome"]]
        ros = imblearn.over_sampling.RandomOverSampler()
        Xtrain, ytrain = ros.fit_resample(Xtrain, ytrain)

        """We must also scale the features X."""
        columns = test_df.columns
        Xtest = test_df.loc[:,test_df.columns!="Outcome"]
        ytest = test_df.loc[:,["Outcome"]]

        scaler = sklearn.preprocessing.StandardScaler().fit(Xtrain)
        scaled_train = scaler.transform(Xtrain)
        scaled_test = scaler.transform(Xtest)

        Xtrain = pd.DataFrame(scaled_train, columns=columns[:-1])
        Xtest = pd.DataFrame(scaled_test, columns=columns[:-1])

        self.featcol = columns[:-1]
        self.outpcol = columns[-1]

        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest

    def k_NN(self):
        """Classify the patientsusing k-NN, selecting the best number of
        neighbours both via a 5-fold and a loo cross-validation procedure.
        Plot the two estimated error for each possible value of k. Add to the
        plot the corresponding test errors (i.e., the test error you would have
        obtained fitting k-NN with the same k) and comment on the results."""
        nnbs = KNeighborsClassifier(n_neighbors=10)
        modl = nnbs.fit(self.Xtrain, self.ytrain)
        score = modl.score(self.Xtest, self.ytest)
        ypred = np.array(modl.predict(self.Xtest).tolist())

        if self.plot:
            yprobas = \
                np.append((1-ypred).reshape(-1,1), ypred.reshape(-1,1), axis=1)
            plot_cumulative_gain(self.ytest.values, yprobas)
            plt.title("Cumulative Gains Curve of kNN prediction with\n"\
                + f"binary accuracy of {100*score:.2f}%")
            plt.show()

        """Research which k in kNN is the best:"""
        num = 100
        klin = np.linspace(1, 100, num)
        scorearr = np.zeros(num)
        for counter, i in enumerate(klin):
            nnbs = KNeighborsClassifier(n_neighbors=int(i))
            modl = nnbs.fit(self.Xtrain, self.ytrain)
            score = modl.score(self.Xtest, self.ytest)
            scorearr[counter] = score

        if self.plot:
            plt.plot(klin, scorearr)
            plt.title("The kNN method analysis for multiple k values.")
            plt.grid()
            plt.xlabel("k")
            plt.ylabel("Prediction scores")
            plt.show()

        return 1

    def GAM2(self):
        """GAM of splines, where we perform variable selection
        to find the best model."""
        from pygam import LogisticGAM, s, l, f
        terms = s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+s(7)

        gam = LogisticGAM(terms=terms, fit_intercept=False)
        mod = gam.gridsearch(self.Xtrain.values, self.ytrain, \
            lam=np.logspace(-3, 3, 11))     # Generate the model
        mod.summary()   # Pseudo-R2: 0.6449
        ypred = mod.predict(self.Xtest)
        MSE1 = np.mean((self.ytest - ypred.reshape(-1,1))**2).values

        if self.plot:
            plt.plot(range(len(ypred.reshape(-1,1))),\
                ypred.reshape(-1,1)-0.5,"r.", label='GAM model')
            plt.plot(range(len(self.ytest)), self.ytest, "b.", label='Testing Data')
            plt.legend()
            plt.title("GAM model with linear terms. Prediction data is\n"\
                + "scaled downwards by 0.5 for visual purposes.")
            plt.ylabel("FFVC score")
            plt.xlabel("Sample no.")
            plt.show()



    def tree_bag_ada(self):
        """Use a classification tree, bagging (both with “probability” and
        “consensus” votes), random forest, neural network and AdaBoost,
        to classify the persons between positive and negative to diabetes."""

        """Classification Tree:"""
        from sklearn import tree
        clfTR = tree.DecisionTreeClassifier()
        clfTR = clfTR.fit(self.Xtrain, self.ytrain)
        predTR = clfTR  # .predict(self.Xtest)

        import sklearn.ensemble
        """Probability Bagging:"""
        clfBG1 = sklearn.ensemble.BaggingClassifier()
        clfBG1.fit(self.Xtrain, self.ytrain)
        predBG1 = clfBG1 #.predict_proba(self.Xtest) # predict using the probas?

        """Consensus Bagging:"""
        clfBG2 = sklearn.ensemble.BaggingClassifier()
        clfBG2.fit(self.Xtrain, self.ytrain)
        predBG2 = clfBG2 #.predict(self.Xtest) # predict using voting/consensus?

        """Random Forest:"""
        from sklearn.ensemble import RandomForestClassifier
        clfRF = RandomForestClassifier(n_estimators=100, max_depth=2,\
            random_state=0)
        clfRF.fit(self.Xtrain, self.ytrain)
        predRF = clfRF #.predict(self.Xtest)

        """Neural Network:"""
        from sklearn.neural_network import MLPClassifier
        clfMLP = MLPClassifier(solver='lbfgs', alpha=1e-5,\
            hidden_layer_sizes=(5, 2), random_state=1)
        clfMLP.fit(self.Xtrain, self.ytrain)
        predNNW = clfMLP #.predict(self.Xtest)

        """AdaBoost:"""
        from sklearn.ensemble import AdaBoostClassifier
        clfADA = AdaBoostClassifier(n_estimators=100, random_state=0)
        clfADA.fit(self.Xtrain, self.ytrain)
        predADA = clfADA #.predict(self.Xtest)

        """Save all the predictions."""
        self.pred_TR    = predTR
        self.pred_BG1   = predBG1
        self.pred_BG2   = predBG2
        self.pred_RF    = predRF
        self.pred_NNW   = predNNW
        self.pred_ADA   = predADA
        return 1

    def best_method(self):
        """Analyze which of these methods produced the best accuracy:"""
        TR_score    = self.pred_TR.score(self.Xtest, self.ytest)
        BG1_score   = self.pred_BG1.score(self.Xtest, self.ytest)
        BG2_score   = self.pred_BG2.score(self.Xtest, self.ytest)
        RF_score    = self.pred_RF.score(self.Xtest, self.ytest)
        NNW_score   = self.pred_NNW.score(self.Xtest, self.ytest)
        ADA_score   = self.pred_ADA.score(self.Xtest, self.ytest)
        print(f"Decision Tree score:\t {TR_score:.4f}")
        print(f"Bagging score:  \t {BG1_score:.4f}")
        print(f"Random Forest score:\t {RF_score:.4f}")
        print(f"Neural Network score:\t {NNW_score:.4f}")
        print(f"ADA Boost score:\t {ADA_score:.4f}")    # ADABoost method
        """Printout:
        -------------------------------
        Decision Tree score:	 0.7165
        Bagging score:  	     0.7244
        Random Forest score:	 0.7441
        Neural Network score:	 0.6417
        ADA Boost score:	     0.7520
        -------------------------------
        The Neural Network performed the best on this data, though all the
        methods had excellent scores. They are all quite capable of predictions.
        """


    def remove_outliers(self):
        """Process the data properly. Removing outliers this time."""

        """
        1 pregnant    : number of pregnancies;
        2 glucose     : plasma glucose concentration at 2 h in an oral glucose
                        tolerance test;
        3 pressure    : diastolic blod pressure (mm Hg);
        4 triceps     : triceps skin fold thickness (mm);
        5 insulin     : 2-h serum insulin (μU/mL);
        6 mass        : body mass index (kg/m 2 );
        7 pedigree    : diabetes pedigree function;
        8 age         : age (years)
        """

        df1 = pd.read_csv("diabetes.csv")
        columns = df1.columns
        """Remove the outliers """
        X = df1.values

        # print(min(X[:,0]))  # minimum pregnant is 0
        # print(max(X[:,0]))  # maximum pregnant is 17.   Both are plausable
        # print(min(X[:,1]))  # minimum glucose is 0
        # print(max(X[:,1]))  # maximum glucose is 199.   Zero is strange.
        # print(min(X[:,2]))  # minimum pressure is 0
        # print(max(X[:,2]))  # maximum pressure is 122.  Zero is strange.
        # print(min(X[:,3]))  # minimum triceps is 0
        # print(max(X[:,3]))  # maximum triceps is 99.    Zero is strange.
        # print(min(X[:,4]))  # minimum insulin is 0
        # print(max(X[:,4]))  # maximum insulin is 846.   Zero is strange.
        # print(min(X[:,5]))  # minimum mass is 0
        # print(max(X[:,5]))  # maximum mass is 67.1.     Zero is strange.
        # print(min(X[:,6]))  # minimum pedigree is 0.078
        # print(max(X[:,6]))  # maximum pedigree is 2.42. Both are plausable
        # print(min(X[:,7]))  # minimum age is 21
        # print(max(X[:,7]))  # maximum age is 81.        Both are plausable
        # print(min(X[:,8]))  # minimum outcome is 0
        # print(max(X[:,8]))  # maximum outcome is 1.     Both are plausable

        """Remove 'Zero' cases for the following categories:"""
        zs = [1, 2, 3, 4, 5]
        for i in zs:
            valid_mask = (X[:,i]>0)
            X = X[valid_mask]

        """Remake the dataframe, and continue as usual."""
        df = pd.DataFrame(X, columns = columns)

        """To illustrate the degree of outlier removal:"""
        # print(df1.shape)    # (768, 9)
        # print(df.shape)     # (392, 9)

        train_df, test_df =\
            sklearn.model_selection.train_test_split(df, test_size=0.33,\
                stratify=df["Outcome"]) # stratify such that the outcome is even

        """Need to upsample the 'ones' cases for the training data"""
        Xtrain = train_df.loc[:,train_df.columns!="Outcome"]
        ytrain = train_df.loc[:,["Outcome"]]
        ros = imblearn.over_sampling.RandomOverSampler()
        Xtrain, ytrain = ros.fit_resample(Xtrain, ytrain)

        """We must also scale the features X."""
        columns = test_df.columns
        Xtest = test_df.loc[:,test_df.columns!="Outcome"]
        ytest = test_df.loc[:,["Outcome"]]

        scaler = sklearn.preprocessing.StandardScaler().fit(Xtrain)
        scaled_train = scaler.transform(Xtrain)
        scaled_test = scaler.transform(Xtest)

        Xtrain = pd.DataFrame(scaled_train, columns=columns[:-1])
        Xtest = pd.DataFrame(scaled_test, columns=columns[:-1])

        self.featcol = columns[:-1]
        self.outpcol = columns[-1]

        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest

        """Redo the exercises 1, 2, 3 using the properly processed data."""
        self.k_NN()
        self.GAM2()
        self.tree_bag_ada()
        self.best_method()

        return 1
