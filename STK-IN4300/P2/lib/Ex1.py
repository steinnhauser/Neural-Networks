import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.model_selection
import sklearn.preprocessing
# from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LassoCV
from sklearn import linear_model
import statsmodels.api as sm


class Exercise1:
    def __init__(self):
        pass

    def scale_data(self):
        """Analysis of the 496 children with 24 features. Takes in argument 'sc'
        which indicates how the data should be split into training/testing data.
        So far takes in arguments sc='50/50' for 50% splitting, sc='index', for
        custom indices to be used for splitting (used in bootstrap and k-fold
        cross validation methods)."""
        df = pd.read_csv('ozone_data.txt', sep = ' ')
        """
        'Column nr.'    'Column name'
        Columns to scale:
            1   ALTER
            9   AGEBGEW
            11  FLGROSS
            12  FMILB
            13  FNOH24
            16  FLTOTMED
            17  FO3H24
            19  FTEH24
            22  FLGEW
            25  FFVC
        One-Hot columns:
            2   ADHEU;
            3   SEX;
            4   HOCHOZON;
            5   AMATOP;
            6   AVATOP;
            7   ADEKZ;
            8   ARAUCH
            10  FSNIGHT
            14  FTIER
            15  FPOLL
            18  FSPT
            20  FSATEM
            21  FSAUGE
            23  FSPFEI
            24  FSHLAUF
        """
        scale_inds = [0, 8, 10, 11, 12, 15, 16, 18, 21, 24]
        oneht_inds = [1, 2, 3, 4, 5, 6, 7, 9, 13, 14, 17, 19, 20, 22, 23]
        """
        Three steps to any Data Preprocessing:
            1. Remove any categorical outliers (assumed to be unnessecary here).
            2. Split the data in training and testing.
            3a. One hot encode categorical data
            3b. Scale the non-categorical data
        """

        train_df, test_df =\
            sklearn.model_selection.train_test_split(df, test_size=0.5)

        """Scaled data processing"""
        col_scale = []
        for c in scale_inds:
            col_scale.append(str(train_df.columns[c]))

        scaler = sklearn.preprocessing.StandardScaler().fit(train_df[col_scale])
        scaled_train = scaler.transform(train_df[col_scale])
        scaled_test = scaler.transform(test_df[col_scale])

        """One hot encoded data processing"""
        col_hot = []
        for h in oneht_inds:
            col_hot.append(str(train_df.columns[h]))

        """If not one-hot, just filter it normally like so:"""
        encoded_train = train_df[col_hot]
        encoded_test = test_df[col_hot]

        """Keep track of the column names:"""
        columns = col_hot+col_scale
        self.columns = columns

        """Combine the two data types"""
        train = np.concatenate((encoded_train, scaled_train), axis=1)
        test = np.concatenate((encoded_test, scaled_test), axis=1)

        train_df = pd.DataFrame(train, columns=columns)
        test_df = pd.DataFrame(test, columns=columns)

        Xtrain  = train_df.loc[:, train_df.columns!='FFVC']
        ytrain  = train_df.loc[:, train_df.columns=="FFVC"]
        Xtest   = test_df.loc[:, test_df.columns!='FFVC']
        ytest   = test_df.loc[:, test_df.columns=="FFVC"]

        """Data is now properly split (50/50) and
        scaled for the first exercise."""
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest  = Xtest
        self.ytest  = ytest
        return 0

    def linear_Gauss(self, plot=True):
        """
        Function to estimate a linear Gaussian regression model to relate
        the forced vital capacity of the indepenent variables.

        Plot the covariance/correlation function k, commonly called the kernel
        of the Gaussian process.

        See Jakel, Scholkopf, and Wichmann, 2007 for more
        """
        Xytrain = np.concatenate((self.Xtrain, self.ytrain), axis=1)
        df = pd.DataFrame(Xytrain, columns=self.columns)
        co = df.corr()  # can call upon 'cov' or 'corr'

        if self.plot:
            ax = sb.heatmap(data=co)
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)    # Fix edges
            plt.title("Correlation Matrix")
            plt.show()

        """Find the covariate with the strongest association with the forced
        vital capacity ("FFVC"). Report the coefficient estimates, their
        standard error, and the associated p-value and comment on them."""

        if self.plot:
            co["FFVC"].plot.bar()
            plt.ylabel("Correlation")
            plt.title("Correlation for all features in relation to 'FFVC'.")
            plt.show()

        maximum = co["FFVC"][:-1].idxmax()  # strongest association
        # print(maximum)  # FLGROSS

        """Report the coefficient estimates, their standard error and the
        associated p-value and comment on them."""

        ols = sm.OLS(self.ytrain, self.Xtrain)
        mod = ols.fit()
        ypred = mod.predict(self.Xtest)

        coeff_est= mod.summary2().tables[1]['Coef.']    # Coefficient estimates
        std_devs = mod.summary2().tables[1]['Std.Err.'] # std of estimates
        p_values = mod.summary2().tables[1]['P>|t|']    # p-values
        MSE = np.mean((self.ytest.values - ypred.values)**2)
        self.LGE1P1 = MSE   # for exercise 7.

        # save these values to the object
        self.OLS_coe = coeff_est
        self.OLS_std = std_devs
        self.OLS_pva = p_values
        self.OLS_fullmse = MSE

        if self.plot:
            barplt = pd.DataFrame(np.c_[coeff_est,std_devs,p_values],\
                index=self.columns[:-1])
            barplt.plot.bar()
            plt.legend(["Coeffs", "std", "p-vals"])
            plt.title("Plot illustrating the Coefficients, Standard\n"+\
                " Deviation and p-values of the linear Gaussian Regression.")
            plt.show()

        return 0

    def backward_elimination(self, stopping_criterion=0.05):
        """Backwards elimination algorithm for feature choice. Typical to set
        significance level p<=0.05. Build model by removing largest p-values"""
        pvals = self.OLS_pva
        sorted_vals = pvals.sort_values(ascending=False)
        sorted_hdrs = sorted_vals.index.tolist()
        accepted_cols = []
        for count, pval in enumerate(sorted_vals):
            if pval<=stopping_criterion:
                accepted_cols.append(sorted_hdrs[count])

        """Find the new coefficients of the features which met the p-value
        criterion using the previous OLS coefficients"""
        new_coefs = self.OLS_coe[self.OLS_coe.index.isin(accepted_cols)]
        return new_coefs

    def forward_selection(self, stopping_criterion=0.05):
        """Forwards selection algorithm for feature choice. Typical for the
        significance p<=0.05. Build the model starting from lowest p-values"""
        pvals = self.OLS_pva
        sorted_vals = pvals.sort_values(ascending=True)
        sorted_hdrs = sorted_vals.index.tolist()
        accepted_cols = []
        for count, pval in enumerate(sorted_vals):
            if pval<=stopping_criterion:
                accepted_cols.append(sorted_hdrs[count])

        """Find the new coefficients of the features which met the p-value
        criterion using the previous OLS coefficients"""
        new_coefs = self.OLS_coe[self.OLS_coe.index.isin(accepted_cols)]
        return new_coefs

    def bf_selection1(self):
        """Compare the Backwards Elimination and Fowards Selection algorithms
        to the original OLS prediction."""
        c1 = 0.05
        c2 = 0.1

        be_coef1 = self.backward_elimination(stopping_criterion = c1)
        be_coef2 = self.backward_elimination(stopping_criterion = c2)
        fs_coef1 = self.forward_selection(stopping_criterion = c1)
        fs_coef2 = self.forward_selection(stopping_criterion = c2)

        """Filter the training matrix to remove all columns except for the ones
        which the methods have deemed worthy."""
        Xtrain_bec1 = \
            self.Xtrain.loc[:,self.Xtrain.columns.isin(be_coef1.index.tolist())]
        Xtrain_bec2 = \
            self.Xtrain.loc[:,self.Xtrain.columns.isin(be_coef2.index.tolist())]
        Xtrain_fsc1 = \
            self.Xtrain.loc[:,self.Xtrain.columns.isin(fs_coef1.index.tolist())]
        Xtrain_fsc2 = \
            self.Xtrain.loc[:,self.Xtrain.columns.isin(fs_coef2.index.tolist())]

        Xtest_bec1 = \
            self.Xtest.loc[:,self.Xtest.columns.isin(be_coef1.index.tolist())]
        Xtest_bec2 = \
            self.Xtest.loc[:,self.Xtest.columns.isin(be_coef2.index.tolist())]
        Xtest_fsc1 = \
            self.Xtest.loc[:,self.Xtest.columns.isin(fs_coef1.index.tolist())]
        Xtest_fsc2 = \
            self.Xtest.loc[:,self.Xtest.columns.isin(fs_coef2.index.tolist())]

        """Recreate exercises 1 and 2 for all of these. Create barplots
        illustrating the coefficients, standard deviations, and p-values for
        each of them. Compare also the MSE values of them all."""

        """Backwards Elimination test 1"""
        ols_bec1 = sm.OLS(self.ytrain, Xtrain_bec1)
        mod_bec1 = ols_bec1.fit()
        ypred_bec1 = mod_bec1.predict(Xtest_bec1)
        self.bec1_coe = mod_bec1.summary2().tables[1]['Coef.']
        self.bec1_std = mod_bec1.summary2().tables[1]['Std.Err.']
        self.bec1_pva = mod_bec1.summary2().tables[1]['P>|t|']
        self.bec1_mse = np.mean((self.ytest.values - ypred_bec1.values)**2)

        """Backwards Elimination test 2"""
        ols_bec2 = sm.OLS(self.ytrain, Xtrain_bec2)
        mod_bec2 = ols_bec2.fit()
        ypred_bec2 = mod_bec2.predict(Xtest_bec2)
        self.bec2_coe = mod_bec2.summary2().tables[1]['Coef.']
        self.bec2_std = mod_bec2.summary2().tables[1]['Std.Err.']
        self.bec2_pva = mod_bec2.summary2().tables[1]['P>|t|']
        self.bec2_mse = np.mean((self.ytest.values - ypred_bec2.values)**2)

        """Forwards Selection test 1"""
        ols_fsc1 = sm.OLS(self.ytrain, Xtrain_fsc1)
        mod_fsc1 = ols_fsc1.fit()
        ypred_fsc1 = mod_fsc1.predict(Xtest_fsc1)
        self.fsc1_coe = mod_fsc1.summary2().tables[1]['Coef.']
        self.fsc1_std = mod_fsc1.summary2().tables[1]['Std.Err.']
        self.fsc1_pva = mod_fsc1.summary2().tables[1]['P>|t|']
        self.fsc1_mse = np.mean((self.ytest.values - ypred_fsc1.values)**2)

        """Forwards Selection test 2"""
        ols_fsc2 = sm.OLS(self.ytrain, Xtrain_fsc2)
        mod_fsc2 = ols_fsc2.fit()
        ypred_fsc2 = mod_fsc2.predict(Xtest_fsc2)
        self.fsc2_coe = mod_fsc2.summary2().tables[1]['Coef.']
        self.fsc2_std = mod_fsc2.summary2().tables[1]['Std.Err.']
        self.fsc2_pva = mod_fsc2.summary2().tables[1]['P>|t|']
        self.fsc2_mse = np.mean((self.ytest.values - ypred_fsc2.values)**2)

        """Save these values for exercise 7."""
        self.BEC1E1P2 = self.bec1_mse
        self.BEC2E1P2 = self.bec2_mse
        self.FSC1E1P2 = self.fsc1_mse
        self.FSC2E1P2 = self.fsc2_mse

        if self.plot:
            """Plot all these in bar plots in the same fashion as before:"""
            barplt = \
                pd.DataFrame(np.c_[self.bec1_coe, self.bec1_std, self.bec1_pva],\
                    index=be_coef1.index)
            barplt.plot.bar()
            plt.legend(["Coeffs", "std", "p-vals"])
            plt.title(f"Plot illustrating the Coefficients, Standard\n"+\
                " Deviation and p-values of Backwards elimination using\n" + \
                    f"the stopping criterion {c1}")

            barplt = \
                pd.DataFrame(np.c_[self.bec2_coe, self.bec2_std, self.bec2_pva],\
                    index=be_coef2.index)
            barplt.plot.bar()
            plt.legend(["Coeffs", "std", "p-vals"])
            plt.title(f"Plot illustrating the Coefficients, Standard\n"+\
                " Deviation and p-values of Backwards elimination using\n" + \
                    f"the stopping criterion {c2}")

            barplt = \
                pd.DataFrame(np.c_[self.fsc1_coe, self.fsc1_std, self.fsc1_pva],\
                    index=fs_coef1.index)
            barplt.plot.bar()
            plt.legend(["Coeffs", "std", "p-vals"])
            plt.title(f"Plot illustrating the Coefficients, Standard\n"+\
                " Deviation and p-values of Forwards selection using\n" + \
                    f"the stopping criterion {c1}")

            barplt = \
                pd.DataFrame(np.c_[self.fsc2_coe, self.fsc2_std, self.fsc2_pva],\
                    index=fs_coef2.index)
            barplt.plot.bar()
            plt.legend(["Coeffs", "std", "p-vals"])
            plt.title(f"Plot illustrating the Coefficients, Standard\n"+\
                " Deviation and p-values of Forwards selection using\n" + \
                    f"the stopping criterion {c2}")

            plt.show()

        """Compare all four models. What model do we expect to perform better?
        Don't need to actually compare them, though it might be good to check"""

        """ We have the following OLS baseline:
        self.OLS_coe
        self.OLS_std
        self.OLS_pva
        self.OLS_fullmse
        """

        return 1

    def bootstrap(self):
        """Bootstrap procedure to find the best (minimize deviance) complexity
        parameter of a lasso regression among a custom grid of points."""
        shuffle_no=100
        alpha_no  =100
        ss = ShuffleSplit(n_splits=shuffle_no, test_size=0.25)
        """Need to preprocess the Xtrain data for each time? Checked the data
        and the means and standard deviations are quite consistenly 0 and 1."""
        alpha_array = np.logspace(0,-7,alpha_no)

        mse_mtx = np.zeros((alpha_no, shuffle_no))
        score_mtx = np.zeros((alpha_no, shuffle_no))

        for c1, a in enumerate(alpha_array):
            for c2, (train_ind, test_ind) in enumerate(ss.split(self.Xtrain)):
                """Loop through multiple bootstrap iterations, updating the training
                and testing indices after each train/test split."""
                data_mtx    = pd.concat([self.Xtrain,self.ytrain], axis=1)
                train_df    = data_mtx[data_mtx.index.isin(train_ind)]
                test_df     = data_mtx[data_mtx.index.isin(test_ind)]

                Xtrain = train_df.loc[:,train_df.columns!="FFVC"]
                ytrain = train_df["FFVC"]
                Xtest = test_df.loc[:,test_df.columns!="FFVC"]
                ytest = test_df["FFVC"]

                clf = linear_model.Lasso(alpha=a, fit_intercept=False)
                clf.fit(Xtrain, ytrain)
                ypred = clf.predict(Xtest)
                mse_mtx[c1,c2] = np.mean((ypred-ytest)**2)
                score_mtx[c1,c2] = clf.score(Xtest, ytest)

            print(f"\rBootstrap {100*c1/alpha_no:.2f}% complete.", end="")
        print("")

        Rscore  = np.mean(score_mtx, axis=1)
        maxind  = np.where(Rscore==max(Rscore))
        maxRscr = Rscore[maxind]
        maxalpha = alpha_array[maxind]

        if self.plot:
            plt.axvline(maxalpha, color='k', linestyle='--', \
                label=r"Optimal $\alpha$")
            plt.plot(alpha_array, Rscore, "ro", label=r"$R^2$ Score data")
            plt.xscale("log")
            plt.xlabel(r"Logarithmic hyperparameter $\alpha$ scale using "+\
                fr"{alpha_no} data points.")
            plt.ylabel(fr"Average $R^2$ score for {shuffle_no} bootstrap splits.")
            plt.title(r"The average $R^2$ scores for bootstrapped data."+ "\n"+\
            fr"The optimal hyperparameter was found to be $\alpha={maxalpha}$")
            plt.grid()
            plt.legend()
            plt.show()

        return maxRscr

    def cross_validation(self):
        """k-fold CV procedure to find the best (minimize deviance) complexity
        parameter of a lasso regression among a custom grid of points."""

        """Need to preprocess the Xtrain data for each time? Checked the data
        and the means and standard deviations are quite consistenly 0 and 1."""
        alpha_no = 100
        alpha_array = np.logspace(0,-7,alpha_no)

        reg = LassoCV(cv = 5, n_jobs = -1, alphas = alpha_array,\
            fit_intercept = False) # 5-fold CV
        reg = reg.fit(self.Xtrain, self.ytrain)

        score = reg.score(self.Xtest, self.ytest)
        return score

    def boot_CV_compare(self):
        """Function to compare the two methods of bootstrap and k-fold
        cross validation for a lasso method."""
        Rscore_boot = self.bootstrap()[0]
        Rscore_kfCV = self.cross_validation()

        """Save these values for exercise 7."""
        self.LASSO1E1P4 = Rscore_boot
        self.LASSO2E1P4 = Rscore_kfCV

        print(f"Bootstrap method accomplished an R² score of {Rscore_boot}")
        print(f"5-fold CV method accomplished an R² score of {Rscore_kfCV}")
        """Printout:
        steinn@SHM-PC:~/Desktop/STK-IN4300/P2$ python3 main.py
        Bootstrap 99.00% complete.
        /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/coordinate_descent.py:1100: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
          y = column_or_1d(y, warn=True)
        Bootstrap method accomplished an R² score of [0.64221808]
        5-fold CV method accomplished an R² score of 0.5668369826165436
        """

        return 1

    def GAM1(self):
        """Generalized Additive Model with possible non-linear effects. Specific
        variables are modelled by splines. Can the possible non-linearities be
        captured by adding polynomial terms to the linear model? Fit such a
        model and comment on the two solutions."""
        from pygam import LinearGAM, s, l, f
        """Non-linear effects are modeled by splines. Analyze the summary table
        and declare which factors should be splined. Do this depending on the
        so-called significance code of the table."""
        terms = l(0)+l(1)+l(2)+l(3)+l(4)+l(5)+l(6)+l(7)+l(8)+l(9)+l(10)+l(11)\
            +l(12)+l(13)+l(14)+l(15)+l(16)+l(17)+l(18)+l(19)+l(20)+l(21)+l(22)\
                +l(23)

        gam = LinearGAM(terms=terms, fit_intercept=False)
        mod = gam.gridsearch(self.Xtrain.values, self.ytrain.values, \
            lam=np.logspace(-3, 3, 11))     # Generate the model
        mod.summary()   # Pseudo-R2: 0.6449
        ypred = mod.predict(self.Xtest)
        MSE1 = np.mean((self.ytest - ypred.reshape(-1,1))**2).values

        if self.plot:
            plt.plot(ypred.reshape(-1,1), label='GAM model')
            plt.plot(self.ytest, label='Testing Data')
            plt.legend()
            plt.title("GAM model with linear terms")
            plt.ylabel("FFVC score")
            plt.xlabel("Sample no.")
            plt.show()

        """Repeat the study adding the 'auto' function, adding splines and
        polynomial contributions."""
        gam = LinearGAM(terms='auto', fit_intercept=False)
        mod = gam.gridsearch(self.Xtrain.values, self.ytrain.values, \
            lam=np.logspace(-3, 3, 11))     # Generate the model
        mod.summary()   # Pseudo-R2: 0.6449
        ypred = mod.predict(self.Xtest)
        MSE2 = np.mean((self.ytest - ypred.reshape(-1,1))**2).values

        if self.plot:
            plt.plot(ypred.reshape(-1,1), label='GAM model')
            plt.plot(self.ytest, label='Testing Data')
            plt.legend()
            plt.title("GAM model with spline terms")
            plt.ylabel("FFVC score")
            plt.xlabel("Sample no.")
            plt.show()

        print(f"Linear GAM produced MSE={MSE1},"+"\n"\
            f"Spline addition produced MSE={MSE2}")

        """Save these values for Exercise 7."""
        self.GAM1E1P5 = MSE1[0]
        self.GAM2E1P5 = MSE2[0]

        return 1

    def comp_boosting(self):
        """Fit a component-wise boosting model, using the models:
            i.      Linear Models
            ii.     Splines
            iii.    Trees
        Report the variables selection frequencies in all three
        cases and the regression coefficients for the first model."""

        """Linear Models:"""
        import sklearn.ensemble as skle
        gbr = skle.GradientBoostingRegressor()
        mod1 = gbr.fit(self.Xtrain, self.ytrain)
        ypred = mod1.predict(self.Xtest) #... fit something.

        ytrue = np.array(self.ytest.values.tolist())
        ypred = ypred.tolist()

        LM_boost_MSE = np.mean((ytrue - ypred)**2)
        """Splines:"""

        # SP_boost_MSE = np.mean((self.ytest - ypred)**2)

        """Trees:"""
        from sklearn.experimental import enable_hist_gradient_boosting
        trr = skle.HistGradientBoostingRegressor()
        mod3 = trr.fit(self.Xtrain, self.ytrain)
        ypred = mod3.predict(self.Xtest)

        ytest = np.array(self.ytest.values.tolist())
        ypred = ypred.tolist()

        TR_boost_MSE = np.mean((ytest - ypred)**2)

        """Save these values for Exercise 7."""
        self.BST1E1P6 = LM_boost_MSE
        self.BST2E1P6 = 0 # SP_boost_MSE
        self.BST3E1P6 = TR_boost_MSE

        """Report the variables selection frequencies in all three cases and
        the regression coefficients for the first model."""

        return 1

    def report_points(self):
        """For each approach, report the training and test error.
        Also comment on them."""
        print("Linear Gauss of Exercise 1:")        # 1 result.
        print(f"\tMSE = {self.LGE1P1:.3f}")
        print("Backwards- and Forwards models of Exercise 2:")  # 4 results.
        print(f"\tMSE of Backwards Elimination 1 and 2:")
        print(f"\t\t MSE1 = {self.BEC1E1P2:.3f} and MSE2 = {self.BEC2E1P2:.3f}")
        print(f"\tMSE of Forwards Selection 1 and 2:")
        print(f"\t\t MSE1 = {self.FSC1E1P2:.3f} and MSE2 = {self.FSC2E1P2:.3f}")
        print("Lasso results of Exercise 4:")       # 1 result.
        print(f"\tBootstrap achieved R² = {self.LASSO1E1P4:.3f}")
        print(f"\t5-fold CV achieved R² = {self.LASSO2E1P4:.3f}")
        print("GAM analysis from Exercise 5:")      # 2 results.
        print(f"\t Linear only model MSE = {self.GAM1E1P5:.3f}")
        print(f"\t Polynomial allowed model MSE = {self.GAM2E1P5:.3f}")
        print("3 boosting models of Exercise 6:")   # 3 results.
        print(f"\ti)   : MSE = {self.BST1E1P6:.3f}")
        print(f"\tii)  : MSE = {self.BST2E1P6:.3f}")
        print(f"\tiii) : MSE = {self.BST3E1P6:.3f}")
        return 1
