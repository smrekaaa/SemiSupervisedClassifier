import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from ssc._semi_supervised_classifier import SemiSupervisedClassifier


class Experiment:
    """ Class for diploma experiment

        Parameters
        ----------
        Attributes
            ----------
            income_eval_df : dataframe
                            Income evaluation data
            meal_demand_df : dataframe
                            Meal demand forecast data
            mobile_price_df : dataframe
                            Mobile price range data
        """

    # Settings for dataframe output
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    def __init__(self):
        self.income_eval_df = self.set_income_evaluation_data()
        self.meal_demand_df = self.set_meal_demand_data()
        self.mobile_price_df = self.set_mobile_price_range_data()

    # INCOME EVALUTATION DATA ------------------------------------------------------------------------------------------

    def set_income_evaluation_data(self):
        """ Setting income evaluation data.

        Returns
        -------
        income_eval_df : dataframe
            income valuation data

            Returns income_eval_df
        """
        income_eval_df = pd.read_csv("data\\income_evaluation.csv", decimal=".", sep=",")  # read data

        # Transforming nominal data to number data types for further processing
        niminal_cols = income_eval_df.select_dtypes(exclude=['float64', 'int64']).columns
        for param in niminal_cols:
            income_eval_df[param] = LabelEncoder().fit_transform(income_eval_df[param])

        return income_eval_df

    def get_income_evaluation_data(self):
        """ Setting and returning input and output data for income_eval_df

        Returns
        -------
        x : dataframe
            input data to be used in classificator
        y : dataframe
            output classes

            Returns x, y.
        """
        cols = self.income_eval_df.columns
        x = self.income_eval_df[cols.drop(' income')]  # Input data, without the output column
        y = self.income_eval_df[' income']  # Output classes

        return x, y

    # MEAL DEMAND DATA -------------------------------------------------------------------------------------------------

    def set_meal_demand_data(self):
        """ Setting meal demand data.

        Returns
        -------
        meal_demand_df : dataframe
            meal demand forecast data

            Returns meal_demand_df
        """
        meal_demand_df = pd.read_csv("data\\meal_demand_train.csv", decimal=".", sep=",", index_col='id')

        # Transforming nominal data to number data types for further processing
        nominal_cols = meal_demand_df.select_dtypes(exclude=['float64', 'int64']).columns
        for param in nominal_cols:
            meal_demand_df[param] = LabelEncoder().fit_transform(meal_demand_df[param])

        meal_demand_df = meal_demand_df.head(40000)
        meal_demand_df.reset_index(inplace=True, drop=True)     # Because of the kfold cross validation
        return meal_demand_df

    def get_meal_demand_data(self):
        """ Setting and returning input and output data for income_eval_df

        Returns
        -------
        x : dataframe
            input data to be used in classificator
        y : dataframe
            output classes

            Returns x, y.
        """
        cols = self.meal_demand_df.columns
        x = self.meal_demand_df[cols.drop('num_orders')]  # Input data, without the output column
        y = self.meal_demand_df['num_orders']  # Output classes

        return x, y

    # MOBILE PRICE RANGE -----------------------------------------------------------------------------------------------

    def set_mobile_price_range_data(self):
        """ Setting mobile price data.

        Returns
        -------
        mobile_price_df : dataframe
                        meal demand forecast data

            Returns meal_demand_df
        """
        mobile_price_df = pd.read_csv("data\\mobile_price_train.csv", decimal=".", sep=",")  # read data

        # Transforming nominal data to number data types for further processing
        nominal_cols = mobile_price_df.select_dtypes(exclude=['float64', 'int64']).columns
        for param in nominal_cols:
            mobile_price_df[param] = LabelEncoder().fit_transform(mobile_price_df[param])

        return mobile_price_df

    def get_mobile_price_range_data(self):
        """ Setting and returning input and output data for mobile_price_df

        Returns
        -------
        x : dataframe
            input data to be used in classificator
        y : dataframe
            output classes

            Returns x, y.
        """
        print(self.mobile_price_df.shape)
        cols = self.mobile_price_df.columns
        x = self.mobile_price_df[cols.drop('price_range')]  # Input data, without the output column
        y = self.mobile_price_df['price_range']  # Output classes

        return x, y

    def make_supervised_classificiations(self, input, output):
        """ Function for testing supervised classification with only labeled data, with k-fold cross validation

            Parameters
            ----------
            input : array-like, shape (n_samples, n_features)
                The input samples.
            output : array-like, shape (n_samples,)
                The target values. An array of int.

            Returns
            -------
            df_res : Dataframe
                Accuracy and f1 score values for every classifier.
            """

        # define Classifier
        classificators_supervised = []
        classificators_supervised.append(KNeighborsClassifier())
        classificators_supervised.append(GaussianNB())
        classificators_supervised.append(DecisionTreeClassifier())

        acc_res = []    # Array of accuracy means for a classifier
        f1_res = []     # Array of f1_score means for a classifier
        classificators = []
        for c in classificators_supervised:

            skf = StratifiedKFold(n_splits=10)  # Stratified KFold for 10 folds

            acc_c = []  # Accuracies
            f1_c = []   # F1 scores
            for train_index, test_index in skf.split(input, output):
                x_train = input.loc[train_index]
                x_test = input.loc[test_index]
                y_train = output[train_index]
                y_test = output[test_index]

                # model
                fitted_c = c.fit(x_train, y_train)
                predictions = fitted_c.predict(x_test)

                acc_c.append(accuracy_score(y_test, predictions))
                f1_c.append(f1_score(y_test, predictions, average='macro'))

            classificators.append(type(c).__name__)
            acc_res.append(np.mean(acc_c))
            f1_res.append(np.mean(f1_c))

        # Dataframe of all the results for visualization
        df_res = pd.DataFrame({})
        df_res['Classificator'] = classificators
        df_res['accuracy'] = acc_res
        df_res['f1_score'] = f1_res
        df_res.set_index('Classificator', inplace=True)
        df_res = df_res.T
        return df_res

    def make_semi_supervised_classificiations(self, input, output):
        """ Function for testing semi-supervised classification with only labeled data

            Parameters
            ----------
            input : array-like, shape (n_samples, n_features)
                The input samples.
            output : array-like, shape (n_samples,)
                The target values. An array of int.

            Returns
            -------
            df_res : Dataframe
                Accuracy and f1 score values for every classifier.
            """

        # define Classifier
        classificators_supervised = []
        classificators_supervised.append(KNeighborsClassifier())
        classificators_supervised.append(GaussianNB())
        classificators_supervised.append(DecisionTreeClassifier())

        acc_res = []    # Array of accuracy means for a classifier
        f1_res = []     # Array of f1_score means for a classifier
        classificators = []  # Array of classifiers' names
        for c in classificators_supervised:

            acc_c = []  # Accuracies
            f1_c = []   # F1_scors

            skf = StratifiedKFold(n_splits=10)  # Stratified KFold for 10 folds

            for train_index, test_index in skf.split(input, output):
                x_train = input.loc[train_index]
                x_test = input.loc[test_index]
                y_train = output[train_index]
                y_test = output[test_index]

                # model
                ssc = SemiSupervisedClassifier(c)
                fitted_c = ssc.fit(X=x_train, y=y_train, _refit=True)
                fitted_c.fit(X=x_train, _refit=False)

                predictions = fitted_c.predict(x_test)
                acc_c.append(accuracy_score(y_test, predictions))
                f1_c.append(f1_score(y_test, predictions, average='macro'))

            classificators.append(type(c).__name__)
            acc_res.append(np.mean(acc_c))
            f1_res.append(np.mean(f1_c))

        #Dataframe of all the results for visualization
        df_res = pd.DataFrame({})
        df_res['Classificator'] = classificators
        df_res['accuray'] = acc_res
        df_res['f1_score'] = f1_res
        df_res.set_index('Classificator', inplace=True)
        df_res = df_res.T
        return df_res

    def make_supervised_classificiations_e2(self, input, output, t):
        """ Function for testing supervised classification with labeled and unlabeled data

            Parameters
            ----------
            input : array-like, shape (n_samples, n_features)
                The input samples.
            output : array-like, shape (n_samples,)
                The target values. An array of int.
            t: double
                Train/Test ratio

            Returns
            -------
            df_res : Dataframe
                Accuracy and f1 score values for every classifier.
            """
        # define Classifier
        classificators_supervised = []
        classificators_supervised.append(KNeighborsClassifier())
        classificators_supervised.append(GaussianNB())
        classificators_supervised.append(DecisionTreeClassifier())

        acc_res = []
        f1_res = []
        classificators = []
        for c in classificators_supervised:

            skf = StratifiedKFold(n_splits=10)  # Stratified KFold for 10 folds

            acc_c = []
            f1_c = []
            for train_index, test_index in skf.split(input, output):
                x_train = input.loc[train_index]
                x_test = input.loc[test_index]
                y_train = output[train_index]
                y_test = output[test_index]

                # split training data to labeled and unlabeled data
                # x_train_1 = labeled train data, x_train_2 = unlabeled train data
                x_train_1, x_train_2, y_train_1, y_test_2 = train_test_split(x_train, y_train,
                                                                             test_size=t,
                                                                             random_state=123)

                # model
                fitted_c = c.fit(x_train_1, y_train_1)
                predictions = fitted_c.predict(x_test)

                acc_c.append(accuracy_score(y_test, predictions))
                f1_c.append(f1_score(y_test, predictions, average='macro'))

            classificators.append(type(c).__name__)
            acc_res.append(np.mean(acc_c))
            f1_res.append(np.mean(f1_c))

        # Dataframe of all the results for visualization
        df_res = pd.DataFrame({})
        df_res['Classificator'] = classificators
        df_res['accuray'] = acc_res
        df_res['f1_score'] = f1_res
        df_res.set_index('Classificator', inplace=True)
        df_res = df_res.T
        return df_res

    def make_semi_supervised_classificiations_e2(self, input, output, t):
        """ Function for testing semi-supervised classification with labeled and unlabeled data

            Parameters
            ----------
            input : array-like, shape (n_samples, n_features)
                The input samples.
            output : array-like, shape (n_samples,)
                The target values. An array of int.
            t: double
                Train/Test ratio

            Returns
            -------
            df_res : Dataframe
                Accuracy and f1 score values for every classifier.
            """

        # define Classifier
        classificators_supervised = []
        classificators_supervised.append(KNeighborsClassifier())
        classificators_supervised.append(GaussianNB())
        classificators_supervised.append(DecisionTreeClassifier())

        # StratifiedKFold object
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'f1_macro': make_scorer(f1_score, average='macro')}

        acc_res = []  # Array of accuracy means for a classifier
        f1_res = []  # Array of f1_score means for a classifier
        classificators = []  # Array of classifiers' names
        for c in classificators_supervised:

            acc_c = []  # Accuracies
            f1_c = []  # F1_scoers

            skf = StratifiedKFold(n_splits=10)  # Stratified KFold for 10 folds

            for train_index, test_index in skf.split(input, output):
                x_train = input.loc[train_index]
                x_test = input.loc[test_index]
                y_train = output[train_index]
                y_test = output[test_index]

                # x_train_1 = labeled train data, x_train_2 = unlabeled train data
                x_train_1, x_train_2, y_train_1, y_train_2 = train_test_split(x_train, y_train,
                                                                             test_size=t,
                                                                             random_state=123)
                # model
                ssc = SemiSupervisedClassifier(c)
                fitted_c = ssc.fit(X=x_train_1, y=y_train_1, _refit=True)
                fitted_c.fit(X=x_train_2, _refit=False)
                predictions = fitted_c.predict(x_test)
                acc_c.append(accuracy_score(y_test, predictions))
                f1_c.append(f1_score(y_test, predictions, average='macro'))

            classificators.append(type(c).__name__)
            acc_res.append(np.mean(acc_c))
            f1_res.append(np.mean(f1_c))

        # Dataframe of all the results for visualization
        df_res = pd.DataFrame({})
        df_res['Classificator'] = classificators
        df_res['accuray'] = acc_res
        df_res['f1_score'] = f1_res
        df_res.set_index('Classificator', inplace=True)
        df_res = df_res.T
        return df_res


if __name__ == '__main__':

    exp = Experiment()

    income_eval_x, income_eval_y = exp.get_income_evaluation_data()  # ie
    meal_demand_x, meal_demand_y = exp.get_meal_demand_data()  # md
    mobile_price_x, mobile_price_y = exp.get_mobile_price_range_data()  # mp

    # EXPERIMENT 1 -----------------------------------------------------------------------------------------------------

    df_ie = exp.make_supervised_classificiations(income_eval_x, income_eval_y)
    df_ie.to_csv("res/df_ie.csv", decimal=".", sep=",", header=True)
    df_mp = exp.make_supervised_classificiations(mobile_price_x, mobile_price_y)
    df_mp.to_csv("res/df_mp.csv", decimal=".", sep=",", header=True)
    df_md = exp.make_supervised_classificiations(meal_demand_x, meal_demand_y)
    df_md.to_csv("res/df_md.csv", decimal=".", sep=",", header=True)

    df_ie_ss = exp.make_semi_supervised_classificiations(income_eval_x, income_eval_y)
    df_ie_ss.to_csv("res/df_ie_ss.csv", decimal=".", sep=",", header=True)
    df_mp_ss = exp.make_semi_supervised_classificiations(mobile_price_x, mobile_price_y)
    df_mp_ss.to_csv("res/df_mp_ss.csv", decimal=".", sep=",", header=True)
    df_md_ss = exp.make_semi_supervised_classificiations(meal_demand_x, meal_demand_y)
    df_md_ss.to_csv("res/df_md_ss.csv", decimal=".", sep=",", header=True)

    # Experiment 2 -----------------------------------------------------------------------------------------------------

    # 25 % labeled training data
    df_ie_2a = exp.make_supervised_classificiations_e2(income_eval_x, income_eval_y, 0.75)
    df_ie_2a.to_csv("res/df_ie_2a.csv", decimal=".", sep=",", header=True)
    df_mp_2a = exp.make_supervised_classificiations_e2(mobile_price_x, mobile_price_y, 0.75)
    df_mp_2a.to_csv("res/df_mp_2a.csv", decimal=".", sep=",", header=True)
    df_md_2a = exp.make_supervised_classificiations_e2(meal_demand_x, meal_demand_y, 0.75)
    df_md_2a.to_csv("res/df_md_2a.csv", decimal=".", sep=",", header=True)

    df_ie_ss_2a = exp.make_semi_supervised_classificiations_e2(income_eval_x, income_eval_y, 0.75)
    df_ie_ss_2a.to_csv("res/df_ie_ss_2a.csv", decimal=".", sep=",", header=True)
    df_mp_ss_2a = exp.make_semi_supervised_classificiations_e2(mobile_price_x, mobile_price_y, 0.75)
    df_mp_ss_2a.to_csv("res/df_mp_ss_2a.csv", decimal=".", sep=",", header=True)
    df_md_ss_2a = exp.make_semi_supervised_classificiations_e2(meal_demand_x, meal_demand_y, 0.75)
    df_md_ss_2a.to_csv("res/df_md_ss_2a.csv", decimal=".", sep=",", header=True)

    # 50 % labeled training data
    df_ie_2b = exp.make_supervised_classificiations_e2(income_eval_x, income_eval_y, 0.5)
    df_ie_2b.to_csv("res/df_ie_2b.csv", decimal=".", sep=",", header=True)
    df_mp_2b = exp.make_supervised_classificiations_e2(mobile_price_x, mobile_price_y, 0.5)
    df_mp_2b.to_csv("res/df_mp_2b.csv", decimal=".", sep=",", header=True)
    df_md_2b = exp.make_supervised_classificiations_e2(meal_demand_x, meal_demand_y, 0.5)
    df_md_2b.to_csv("res/df_md_2b.csv", decimal=".", sep=",", header=True)

    df_ie_ss_2b = exp.make_semi_supervised_classificiations_e2(income_eval_x, income_eval_y, 0.5)
    df_ie_ss_2b.to_csv("res/df_ie_ss_2b.csv", decimal=".", sep=",", header=True)
    df_mp_ss_2b = exp.make_semi_supervised_classificiations_e2(mobile_price_x, mobile_price_y, 0.5)
    df_mp_ss_2b.to_csv("res/df_mp_ss_2b.csv", decimal=".", sep=",", header=True)
    df_md_ss_2b = exp.make_semi_supervised_classificiations_e2(meal_demand_x, meal_demand_y, 0.5)
    df_md_ss_2b.to_csv("res/df_md_ss_2b.csv", decimal=".", sep=",", header=True)

    # 75 % labeled training data
    df_ie_2c = exp.make_supervised_classificiations_e2(income_eval_x, income_eval_y, 0.25)
    df_ie_2c.to_csv("res/df_ie_2c.csv", decimal=".", sep=",", header=True)
    df_mp_2c = exp.make_supervised_classificiations_e2(mobile_price_x, mobile_price_y, 0.25)
    df_mp_2c.to_csv("res/df_mp_2c.csv", decimal=".", sep=",", header=True)
    df_md_2c = exp.make_supervised_classificiations_e2(meal_demand_x, meal_demand_y, 0.25)
    df_md_2c.to_csv("res/df_md_2c.csv", decimal=".", sep=",", header=True)

    df_ie_ss_2c = exp.make_semi_supervised_classificiations_e2(income_eval_x, income_eval_y, 0.25)
    df_ie_ss_2c.to_csv("res/df_ie_ss_2c.csv", decimal=".", sep=",", header=True)
    df_mp_ss_2c = exp.make_semi_supervised_classificiations_e2(mobile_price_x, mobile_price_y, 0.25)
    df_mp_ss_2c.to_csv("res/df_mp_ss_2c.csv", decimal=".", sep=",", header=True)
    df_md_ss_2c = exp.make_semi_supervised_classificiations_e2(meal_demand_x, meal_demand_y, 0.25)
    df_md_ss_2c.to_csv("res/df_md_ss_2c.csv", decimal=".", sep=",", header=True)




