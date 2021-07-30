import numpy as np
import warnings

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
        classifier_ : trained classifier from base_estimator
        X_lab_: array-like, shape (n_samples, n_features)
                The enlarged labeled training dataset.
        X_unlab_: array-like, shape (m_samples, m_features)
                The unlabeled training input samples.
        X_new_: array-like, shape (n_samples+m_samples, n_features+m_features)
                The unlbeled+labeled training input samples.
        y_lab_: array-like, shape (n_samples,),
                The enlarged labeled target values. An array of int.
        y_unlab_: array-like, shape (n_samples,),
                  The labeled target values gotten with prediction of X_unlabeled data and base_classifier . An array of int.
        y_new_:
    """

    def __init__(self, base_estimator):
        """ Init function

            Parameters
            ----------
            base_estimator : base sci-kit classifier that will be used
            """
        # Check if base_estimator is given as a class
        if is_classifier(base_estimator):
            self.base_estimator = base_estimator
            self.n_add = 0
        else:
            raise ValueError('"base_estimator" must be a classifier')

    def fit(self, X, y=None, _refit=True):
        """Fitting function for a classifier.

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

        # If _refit=True -> train classifier and save it
        if _refit:
            self.X_lab_ = self.X_
            self.y_lab_ = self.y_
            self.classifier_ = self.base_estimator.fit(self.X_lab_, self.y_lab_)
            self.classes = self.classifier_.classes_

        # If _refit=False ->
            # Iterativly -> end when there is no more data with >95% prob. conf. or no data left
            # 1. classify X with base_estimator
            # Get confidence probabilities
            # Calibrate probabilitites
            # sort newData by confidence
            # add only data with 95% confidence

            # 2. use newData+oldData for training new clissifier
            # Save only new classifier
        else:
            self.X_unlab_ = self.X_

            # tu se začne zanka!

            # safe break
            i = 0
            while self.X_unlab_.size != 0 and i <= 400:
                i = i+1

                print("x_lab_ len: " + str(len(self.X_lab_)))
                # 1. calibrate classifier to get predit_proba() method
                self.classifier_ = CalibratedClassifierCV(self.classifier_, cv='prefit')
                self.classifier_.fit(self.X_lab_, self.y_lab_)

                # Predict y with calibrated_classifier
                predictions = self.base_estimator.predict(self.X_unlab_)
                print("Predictions")
                print(predictions[:10])

                # Get predicition probabilities
                probs = self.classifier_.predict_proba(self.X_unlab_)
                print("Probs:")
                print(len(probs))

                # Create a dataframe and merge predictions and probabilities toghether
                df = pd.DataFrame([])
                df['prediction'] = predictions
                df['probs'] = list(map(max, probs))
                df['id'] = df.index     # redundant
                print(df.head())

                # Calculate number of rows added to labeled dataset (5% of data)
                # self.n_add = np.math.floor(len(predictions) * 0.05)

                # get only rows where prediciton probabilities are higher than 80 %
                df_max = df.loc[df['probs'] >= 0.80]

                # If df_max is empty break out of the loop, because there is nothing to
                if df_max.empty:
                    print("break initeration: " + str(i))
                    break
                #df_max_add.sort_values(by=['probs'], ascending=False, inplace=True)
                #df_max_add = df_max.head(self.n_add)
                #print(df_max_add.head())

                X_new_ = []
                y_new_ = []
                for index, row in df_max.iterrows():
                    X_new_.append(self.X_unlab_[index])
                    y_new_.append(predictions[index])
                    # noinspection PyTypeChecker
                    np.delete(self.X_unlab_, index, 0)
                print("x_unlab_ len: " + str(len(self.X_lab_)))

            # for index, p in enumerate(predictions):
            #     if probs[index][p] >= 0.95:
            #         self.X_new_.append(self.X_unlab_[index])
            #         self.y_new_.append(p)

            #print(self.X_new_[:10])
            # print(self.y_new_)

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

        # Return the classifier
        return self

        #def self_training(self, x, y):

# Personal test
if __name__ == '__main__':

    # define dataset
    X, y = make_classification(n_samples=10000, n_features=6, n_informative=2, n_redundant=0, random_state=1)
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
    # split train into labeled and unlabeled
    X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50,
                                                                            random_state=1, stratify=y_train)
    # summarize training set size
    print('Labeled Train Set:', X_train_lab.shape, y_train_lab.shape)
    print('Unlabeled Train Set:', X_test_unlab.shape, y_test_unlab.shape)
    # summarize test set size
    print('Test Set:', X_test.shape, y_test.shape)

    # define Classifier
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
    trained_classifier = ssc_knc.fit(X=X_train_lab, y=y_train_lab, _refit=True)
    print(trained_classifier.__class__)

    # Semi-supervised classification: Use trained classifier for unlabeled data
    ssc_knc.fit(X=X_test_unlab, _refit=False)