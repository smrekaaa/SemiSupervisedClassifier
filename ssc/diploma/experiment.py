import warnings
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, f1_score

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate

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
    # Settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    warnings.filterwarnings("ignore")

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
        #df2_center = pd.read_csv("data\\fulfilment_center_info.csv", decimal=".", sep=",", index_col='center_id')
        #df2_meal = pd.read_csv("data\\meal_info.csv", decimal=".", sep=",", index_col='meal_id')

        # Merge dataframes by center_id, meal_id
        #meal_demand_df = pd.DataFrame.merge(meal_demand_df, df2_center, on='center_id')
        #meal_demand_df = pd.DataFrame.merge(meal_demand_df, df2_meal, on='meal_id')

        # Transforming nominal data to number data types for further processing
        nominal_cols = meal_demand_df.select_dtypes(exclude=['float64', 'int64']).columns
        for param in nominal_cols:
            meal_demand_df[param] = LabelEncoder().fit_transform(meal_demand_df[param])

        return meal_demand_df.head(40000)

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



if __name__ == '__main__':

    exp = Experiment()

    income_eval_x, income_eval_y = Experiment.get_income_evaluation_data(exp)  # ie
    meal_demand_x, meal_demand_y = Experiment.get_meal_demand_data(exp)  # md
    mobile_price_x, mobile_price_y = Experiment.get_mobile_price_range_data(exp)  # mp

    # # define Classifier
    # knc = KNeighborsClassifier()
    # nbc = GaussianNB()
    # dtc = DecisionTreeClassifier()
    #
    # # Define semi-supervised classifier
    # ssc_knc = SemiSupervisedClassifier(knc)
    # ssc_nbc = SemiSupervisedClassifier(nbc)
    # ssc_dtc = SemiSupervisedClassifier(dtc)

    # _refit=True & y=None
    # returns error
    # ssc_knc.fit(X_train_lab, _refit=True)

    # # Train classifier with labeled data
    # trained_classifier = ssc_knc.fit(X=ie_X_train_lab, y=ie_y_train_lab, _refit=True)
    # print(trained_classifier.__class__)
    #
    # # Semi-supervised classification: Use trained classifier for unlabeled data
    # ssc_knc = ssc_knc.fit(ie_X_test_unlab, _refit=False)
    #
    # # Predict test data
    # res1 = ssc_knc.predict(ie_X_test)
    #
    # # Accuracy
    # acc = metrics.accuracy_score(ie_y_test, res1)
    # f1 = metrics.f1_score(ie_y_test, res1, average=None)
    # print("Accuracy: " + str(acc))
    # print("FS: " + str(f1))
    #
    # t1 = KNeighborsClassifier(n_neighbors=5)
    # t1 = t1.fit(X=ie_X_train_lab, y=ie_y_train_lab)
    # res2 = t1.predict(ie_X_test)
    # acc2 = metrics.accuracy_score(ie_y_test, res2)
    # f2 = metrics.f1_score(ie_y_test, res2, average=None)
    # print("Accuracy 2: " + str(acc2))
    # print("FS 2: " + str(f2))

    # # Train classifier with labeled data
    # trained_classifier = ssc_knc.fit(X=md_X_train_lab, y=md_y_train_lab, _refit=True)
    # print(trained_classifier.__class__)
    #
    # # Semi-supervised classification: Use trained classifier for unlabeled data
    # ssc_knc = ssc_knc.fit(md_X_test_unlab, _refit=False)
    #
    # # Predict test data
    # res1 = ssc_knc.predict(md_X_test)
    #
    # # Accuracy
    # acc = metrics.accuracy_score(md_y_test, res1)
    # f1 = metrics.f1_score(md_y_test, res1, average=None)
    # print("Accuracy: " + str(acc))
    # print("FS: " + str(f1))
    #
    # t1 = KNeighborsClassifier(n_neighbors=5)
    # t1 = t1.fit(X=md_X_train_lab, y=md_y_train_lab)
    # res2 = t1.predict(md_X_test)
    # acc2 = metrics.accuracy_score(md_y_test, res2)
    # f2 = metrics.f1_score(md_y_test, res2, average=None)
    # print("Accuracy 2: " + str(acc2))
    # print("FS 2: " + str(f2))

    # EXPERIMENT 1 -----------------------------------------------------------------------------------------------------

    def make_supervised_classificiations(input, output):

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
                # print('{} of KFold {}'.format(i, skf.n_splits))
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
        df_res['Classificators'] = classificators
        df_res['accuray'] = acc_res
        df_res['f1_score'] = f1_res

        print("done")
        return df_res

    df_ie = make_supervised_classificiations(income_eval_x, income_eval_y)
    # df_mp = make_supervised_classificiations(mobile_price_x, mobile_price_y)
    # df_md = make_supervised_classificiations(meal_demand_x, meal_demand_y)

    print(df_ie)

    def make_semi_supervised_classificiations(input, output):

        # define Classifier
        classificators_supervised = []
        classificators_supervised.append(KNeighborsClassifier())
        classificators_supervised.append(GaussianNB())
        classificators_supervised.append(DecisionTreeClassifier())

        # StratifiedKFold object
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'f1_macro': make_scorer(f1_score, average='macro')}

        acc_res = []    # Array of accuracy means for a classifier
        f1_res = []     # Array of f1_score means for a classifier
        classificators = []  # Array of classifiers' names
        for c in classificators_supervised:

            acc_c = []  # Accuracies
            f1_c = []   # F1_scoers

            skf = StratifiedKFold(n_splits=10)  # Stratified KFold for 10 folds

            for train_index, test_index in skf.split(input, output):
                #print('{} of KFold {}'.format(i, skf.n_splits))
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
        df_res['Classificators'] = classificators
        df_res['accuray'] = acc_res
        df_res['f1_score'] = f1_res
        print("done")
        return df_res

    df_ie_ss = make_semi_supervised_classificiations(income_eval_x, income_eval_y)
    # df_mp_ss = make_semi_supervised_classificiations(mobile_price_x, mobile_price_y)
    # df_md_ss = make_semi_supervised_classificiations(meal_demand_x, meal_demand_y)

    print(df_ie_ss)

    # Experiment 2 -----------------------------------------------------------------------------------------------------

    # Income evaluation
    # split into train and test
    # ie_X_train, ie_X_test, ie_y_train, ie_y_test = train_test_split(income_eval_x, income_eval_y,
    #                                                                 test_size=0.25, random_state=123)
    # # split train into labeled and unlabeled
    # ie_X_train_lab, ie_X_test_unlab, ie_y_train_lab, ie_y_test_unlab = train_test_split(ie_X_train, ie_y_train,
    #                                                                                     test_size=0.25,
    #                                                                                     random_state=123)
    # # Meal demand
    # md_X_train, md_X_test, md_y_train, md_y_test = train_test_split(meal_demand_x, meal_demand_y,
    #                                                                 test_size=0.25, random_state=123)
    #
    # md_X_train_lab, md_X_test_unlab, md_y_train_lab, md_y_test_unlab = train_test_split(md_X_train, md_y_train,
    #                                                                                     test_size=0.25,
    #                                                                                     random_state=123)
    # # Mobile price
    # mp_X_train, mp_X_test, mp_y_train, mp_y_test = train_test_split(mobile_price_x, mobile_price_y,
    #                                                                 test_size=0.25, random_state=123)
    #
    # mp_X_train_lab, mp_X_test_unlab, mp_y_train_lab, mp_y_test_unlab = train_test_split(mp_X_train, mp_y_train,
    #                                                                                     test_size=0.25,
    #                                                                                     random_state=123)

    def make_supervised_classificiations_e2(input, output, t):

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
                # print('{} of KFold {}'.format(i, skf.n_splits))
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
        df_res['Classificators'] = classificators
        df_res['accuray'] = acc_res
        df_res['f1_score'] = f1_res

        print("done")
        return df_res

    df_ie_2 = make_supervised_classificiations_e2(income_eval_x, income_eval_y, 0.75)
    # df_mp_2 = make_supervised_classificiations_e2(mobile_price_x, mobile_price_y)
    # df_md_2 = make_supervised_classificiations_e2(meal_demand_x, meal_demand_y)

    print(df_ie_2)

    def make_semi_supervised_classificiations_e2(input, output, t):

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
                # print('{} of KFold {}'.format(i, skf.n_splits))

                x_train = input.loc[train_index]
                x_test = input.loc[test_index]
                y_train = output[train_index]
                y_test = output[test_index]

                # x_train_1 = labeled train data, x_train_2 = unlabeled train data
                x_train_1, x_train_2, y_train_1, y_test_2 = train_test_split(x_train, y_train,
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
        df_res['Classificators'] = classificators
        df_res['accuray'] = acc_res
        df_res['f1_score'] = f1_res
        print("done")
        return df_res

    # df_ie_ss_2 = make_supervised_classificiations_e2(income_eval_x, income_eval_y, 0.75)
    # df_mp_ss_2 = make_semi_supervised_classificiations_e2(mobile_price_x, mobile_price_y)
    # df_md_ss_2 = make_semi_supervised_classificiations_e2(meal_demand_x, meal_demand_y)