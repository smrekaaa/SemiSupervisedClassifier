import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, is_classifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

# Test Program
from sklearn.neighbors import KNeighborsClassifier
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
                The labeled training input samples.
        X_unlab_: array-like, shape (m_samples, m_features)
                The unlabeled training input samples.
        X_new_: array-like, shape (n_samples+m_samples, n_features+m_features)
                The unlbeled+labeled training input samples.
        y_lab_: array-like, shape (n_samples,),
                The labeled target values. An array of int.
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
            raise ValueError(' Parameter "y" was not given!')

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

        # If _refit=False ->
            # 1. classify X with base_estimator
            # 2. use newData+oldData for training new clissifier
            # Save only new classifier
        else:
            self.X_unlab_ = self.X_

            # Classify X with base_estimator
            self.y_unlab_ = self.base_estimator.predict(self.X_unlab_)

            self.X_new_ = np.concatenate((self.X_lab_, self.X_unlab_))
            self.y_new_ = np.concatenate((self.y_lab_, self.y_unlab_))
            self.classifier_ = self.base_estimator.fit(self.X_new_, self.y_new_)

            # Check values
            print('X_: {}'.format(self.X_.shape))
            print('X_lab_: {}'.format(self.X_lab_.shape))
            print('X_unlab_: {}'.format(self.X_unlab_.shape))
            print('X_new_: {}'.format(self.X_new_.shape))

            print('y_: {}'.format(self.y_.shape))
            print('y_lab_: {}'.format(self.y_lab_.shape))
            print('y_unlab_: {}'.format(self.y_unlab_.shape))
            print('y_new_: {}'.format(self.y_new_.shape))

        # Return the classifier
        return self


# Personal test
if __name__ == '__main__':

    # define dataset
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1)
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

    # Define semi-supervised classifier
    ssc_knc = SemiSupervisedClassifier(knc)

    # _refit=True & y=None
    # returns error
    # ssc_knc.fit(X_train_lab, _refit=True)

    # Train classifier with labeled data
    trained_classifier = ssc_knc.fit(X=X_train_lab, y=y_train_lab, _refit=True)
    print(trained_classifier.__class__)

    # Semi-supervised classification: Use trained classifier for unlabeled data
    ssc_knc.fit(X=X_test_unlab, _refit=False)