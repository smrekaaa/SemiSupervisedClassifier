import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

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
        df2_center = pd.read_csv("data\\fulfilment_center_info.csv", decimal=".", sep=",", index_col='center_id')
        df2_meal = pd.read_csv("data\\meal_info.csv", decimal=".", sep=",", index_col='meal_id')

        # Merge dataframes by center_id, meal_id
        meal_demand_df = pd.DataFrame.merge(meal_demand_df, df2_center, on='center_id')
        meal_demand_df = pd.DataFrame.merge(meal_demand_df, df2_meal, on='meal_id')

        # Transforming nominal data to number data types for further processing
        nominal_cols = meal_demand_df.select_dtypes(exclude=['float64', 'int64']).columns
        for param in nominal_cols:
            meal_demand_df[param] = LabelEncoder().fit_transform(meal_demand_df[param])

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
        cols = self.mobile_price_df.columns
        x = self.mobile_price_df[cols.drop('price_range')]  # Input data, without the output column
        y = self.mobile_price_df['price_range']  # Output classes

        return x, y


if __name__ == '__main__':
    exp = Experiment()

    income_eval_x, income_eval_y = Experiment.get_income_evaluation_data(exp)  # ie
    meal_demand_x, meal_demand_y = Experiment.get_meal_demand_data(exp)  # md
    mobile_price_x, mobile_price_y = Experiment.get_meal_demand_data(exp)  # mp

    # Income evaluation
    # split into train and test
    ie_X_train, ie_X_test, ie_y_train, ie_y_test = train_test_split(income_eval_x, income_eval_y,
                                                                    test_size=0.50, random_state=123)
    # split train into labeled and unlabeled
    ie_X_train_lab, ie_X_test_unlab, ie_y_train_lab, ie_y_test_unlab = train_test_split(ie_X_train, ie_y_train,
                                                                                        test_size=0.50,
                                                                                        random_state=123)
    # Meal demand
    md_X_train, md_X_test, md_y_train, md_y_test = train_test_split(meal_demand_x, meal_demand_y,
                                                                    test_size=0.50, random_state=123)

    md_X_train_lab, md_X_test_unlab, md_y_train_lab, md_y_test_unlab = train_test_split(md_X_train, md_y_train,
                                                                                        test_size=0.50,
                                                                                        random_state=123)
    # Mobile price
    mp_X_train, mp_X_test, mp_y_train, mp_y_test = train_test_split(mobile_price_x, mobile_price_y,
                                                                    test_size=0.50, random_state=123)

    mp_X_train_lab, mp_X_test_unlab, mp_y_train_lab, mp_y_test_unlab = train_test_split(mp_X_train, mp_y_train,
                                                                                        test_size=0.50,
                                                                                        random_state=123)

    # # define Classifier
    knc = KNeighborsClassifier()
    nbc = GaussianNB()
    dtc = DecisionTreeClassifier()

    # Define semi-supervised classifier
    ssc_knc = SemiSupervisedClassifier(knc)
    ssc_nbc = SemiSupervisedClassifier(nbc)
    ssc_dtc = SemiSupervisedClassifier(dtc)

    # _refit=True & y=None
    # returns error
    # ssc_knc.fit(X_train_lab, _refit=True)

    # Train classifier with labeled data
    trained_classifier = ssc_knc.fit(X=ie_X_train_lab, y=ie_y_train_lab, _refit=True)
    print(trained_classifier.__class__)

    # Semi-supervised classification: Use trained classifier for unlabeled data
    ssc_knc.fit(ie_X_test_unlab, _refit=False)

# CLASSIFICATION ---------------------------------------------------------------------------------------------------
#
# # split into train and test
# X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df1_x, df1_y, test_size=0.50,
#                                                             random_state=1, stratify=df1_y)
# # split train into labeled and unlabeled
# X_train_lab_1, X_test_unlab_1, y_train_lab_1, y_test_unlab_1 = train_test_split(X_train_1, y_train_1,
#                                                                                 test_size=0.50, random_state=1,
#                                                                                 stratify=y_train_1)
# # summarize training set size
# print('Labeled Train Set:', X_train_lab_1.shape, y_train_lab_1.shape)
# print('Unlabeled Train Set:', X_test_unlab_1.shape, y_test_unlab_1.shape)
# # summarize test set size
# print('Test Set:', X_test_1.shape, y_test_1.shape)
#
# # define Classifier
# knc = KNeighborsClassifier()
# nbc = GaussianNB()
# dtc = DecisionTreeClassifier()
#
# # Define semi-supervised classifier
# ssc_knc = SemiSupervisedClassifier(knc)
# ssc_nbc = SemiSupervisedClassifier(nbc)
# ssc_dtc = SemiSupervisedClassifier(dtc)
#
# # _refit=True & y=None
# # returns error
# # ssc_knc.fit(X_train_lab, _refit=True)
#
# # Train classifier with labeled data
# trained_classifier = ssc_knc.fit(X=X_train_lab_1, y=y_train_lab_1, _refit=True)
# print(trained_classifier.__class__)
#
# # Semi-supervised classification: Use trained classifier for unlabeled data
# ssc_knc.fit(X=X_test_unlab_1, _refit=False)
