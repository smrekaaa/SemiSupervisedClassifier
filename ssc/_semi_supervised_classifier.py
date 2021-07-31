import time

import numpy as np
import warnings

import joblib
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, is_classifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd

# Test Program
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class SemiSupervisedClassifier(ClassifierMixin, BaseEstimator):
    """ Base class for semi-supervised classifier

    Parameters
    ----------
    Attributes
        ----------
        base_estimator_ :   sci-kit base_estimator
                            untrained classifier from base_estimator
        classifier_ :   sci-kit base_estimator
                        trained classifier from base_estimator
        X_lab_: array-like, shape (n_samples, n_features)
                The enlarged labeled training dataset.
        X_:     array-like, shape (n_samples, n_features)
                Validated labeled dataset.
        X_unlab_: array-like, shape (m_samples, m_features)
                The unlabeled training input samples.
        y_lab_: array-like, shape (n_samples,),
                The enlarged labeled target values. An array of int.
        y_:     array-like, shape (n_samples,),
                Validated labeled target values. An array of int.
        y_unlab_: array-like, shape (n_samples,),
                  The labeled target values gotten with prediction of X_unlabeled data and base_classifier . An array of int.
        fitted_:    boolean, default=False
                    tells if classifier_ was already fitted.
        self_trained_:  boolean, default=False
                        tells if classifier_ was already self-trained.
        min_p_:     double, range(0.0, 1.00), default=0.5
                    Minimum prediction probability limit in self-training process
        max_p_:     double, range(0.0, 1.00), default=0.9
                    Starting prediction probability limit in self-training process
        p_step_:     double, range(0.0, 1.0), default=0.1, < max_p_
                    Step value for going from max_p to min_p
    """

    def __init__(self, base_estimator):
        """ Init function

            Parameters
            ----------
            base_estimator : base sci-kit classifier that will be used
            """

        # Check if base_estimator is given as a class
        if is_classifier(base_estimator):
            self.base_estimator_ = base_estimator
            self.fitted_ = False
            self.self_trained_ = False
            self.min_p_ = 0.8
            self.max_p_ = 0.9
            self.p_step_ = 0.1
            self.classes_ = 0.1


        else:
            raise ValueError('"base_estimator" must be a classifier')

    def fit(self, X, y=None, _refit=True):
        """Fitting function for a classifier. For semi-supervised learning, first fit with _refit=False needs to be called

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : null or array-like, shape (n_samples,), default=none
            The target values. An array of int.
        _refit: boolean, True or False, default=True
                Whether train the same classifier again or
                continue with semi-supervised learning

        Returns
        -------
        self : object
            Returns self.
        """

        # If _refit=True and y is None -> return error
        if _refit and y is None:
            raise ValueError('Parameter "y" was not given!')

        # If _refit=False and y not None -> return warning (y will be ignored)
        if not _refit and y is not None:
            warnings.warn('Parameter "y" will be ignored, because _refit is False',
                          category=UserWarning)

        # Validate and convert X
        # X_: The converted and validated array of X
        self.X_ = check_array(X,
                              accept_sparse=True)

        # Validate and convert y if not None
        # y_: The converted and validated array of y
        if y is not None:
            self.y_ = check_array(y,
                                  ensure_2d=False)

        # If _refit=True
        if _refit:
            self.X_lab_ = self.X_
            self.y_lab_ = self.y_

            # train classifier and save it
            self.classifier_ = self.base_estimator_.fit(self.X_lab_, self.y_lab_)
            # self.classes_ = np.unique(self.y_)
            self.classes_ = self.classifier_.classes_
            self.fitted_ = True

        # If _refit=False
        else:

            # Check if X_lab_ and y_lab_ were given and classifier_ fitted
            if not self.fitted_:
                raise ValueError('No labeled X an y were given. First run fit() with refit_=True.')

            self.X_unlab_ = self.X_

            # Iterative process for classifier self-training
            i = 0
            while self.X_unlab_.size != 0 and i <= 400:
                i = i+1

                print("x_lab_ len: " + str(len(self.X_lab_)))

                # Calibrate classifier to get predit_proba() method and refit it
                # method='isotonic' - For data with more than 1000 items
                self.classifier_ = CalibratedClassifierCV(self.classifier_, cv='prefit')
                self.classifier_.fit(self.X_lab_, self.y_lab_)

                # Get predictions for y with calibrated classifier_
                predictions = self.base_estimator_.predict(self.X_unlab_)

                # Get predicition probabilities
                probs = self.classifier_.predict_proba(self.X_unlab_)
                print("Probs:")
                print(len(probs))

                # Create a dataframe and merge predictions and probabilities together
                df = pd.DataFrame([])
                df['prediction'] = predictions

                # Gets the highest probability in the array which has the same index as prediciton class
                df['probs'] = list(map(max, probs))

                # Sort dataframe by probabilities from the highest to lowest
                df.sort_values(by=['probs'], ascending=False, inplace=True)
                #print(df.head())

                df_max = pd.DataFrame({})   # Dataframe of the highest probabilities rows of 'df'

                # get only rows where prediction probabilities are higher than p
                # min_p is set to 0.5 by default, because less would not make sense
                p = self.max_p_  # starting probability value
                while df_max.empty and p > self.min_p_:
                    df_max = df.loc[df['probs'] >= p]
                    p = p - self.p_step_

                # If df_max is empty return self, because there is nothing to classify anymore
                if df_max.empty:
                    print("break initeration: " + str(i))
                    self.self_trained_ = True
                    return self

                print("df_max shape: " + str(df_max.shape))

                X_new_ = []     # temporary array of newly classified inputs
                y_new_ = []     # temporary array of predictions with high probability
                i_del = []      # array of indexes to be deleted from unlabeled input set

                # Itterating over data with the highest prediction probabilities
                for index, row in df_max.iterrows():
                    X_new_.append(self.X_unlab_[index])
                    y_new_.append(predictions[index])
                    i_del.append(index)

                self.X_unlab_ = np.delete(self.X_unlab_, i_del, 0)  # Delete all newly classified inputs
                # print("x_unlab_ len: " + str(len(self.X_unlab_)))

                # Contenate newly labeled data with old labeled data
                self.X_lab_ = np.concatenate((self.X_lab_, X_new_))
                self.y_lab_ = np.concatenate((self.y_lab_, y_new_))

            # Check values
            # print('X_: {}'.format(self.X_.shape))
            # print('X_lab_: {}'.format(self.X_lab_.shape))
            # print('X_unlab_: {}'.format(self.X_unlab_.shape))
            # print('X_new_: {}'.format(self.X_new_.shape))
            #
            # print('y_: {}'.format(self.y_.shape))
            # print('y_lab_: {}'.format(self.y_lab_.shape))
            # print('y_unlab_: {}'.format(self.y_unlab_.shape))
            # print('y_new_: {}'.format(self.y_new_.shape))

        self.self_trained_ = True

        # Return the classifier
        return self

    def predict(self, X):
        """Predict data function

        Parameters
        ----------
        X : array-like, shape (n_samples)
            The input samples.

        Returns
        -------
        self : array-like, shape (n_samples)
            Returns predictions.
        """

        # Check if classifier was selftrained
        if not self.self_trained_:
            raise ValueError("Classifier needs to cal fit() first.")

        # Validate and convert X
        X = check_array(X,
                        accept_sparse=True)

        return self.classifier_.predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        return super().score(X, y, sample_weight)

    def get_params(self, deep=False):
        """ Get some class paramaters: min_p, max_p_, p_step_

           Returns
           -------
           {} : dictionary,
               Returns parameters.
           """
        return {
                "base_estimator": self.base_estimator_,
                # "fitted_": self.fitted_,
                # "self_trained_": self.self_trained_,
                # "min_p_": self.min_p_,
                # "max_p_": self.max_p_,
                # "p_step_": self.p_step_,
                # "classes_": self.classes_,
        }

    def set_params(self, **parameters):
        """ Sets some class paramaters: min_p, max_p_, p_step_

           Returns
           -------
           self :
                Returns salf.
           """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self


