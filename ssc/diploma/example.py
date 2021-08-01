from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from ssc._semi_supervised_classifier import SemiSupervisedClassifier

if __name__ == '__main__':

    # Data input-X and output-y (big enough)
    X, y = make_classification(n_samples=10000,
                               n_features=5,
                               n_informative=5,
                               n_redundant=0,
                               n_classes=3)

    # Training and test data sets
    x_train, x_test, y_train, y_test = train_test_split( X, y,
                                                         test_size=0.75,
                                                         random_state=123)

    # Labeled and "unlabeled" data sets
    x_lab, x_unlab, y_lab, y_unlab = train_test_split(x_train, y_train,
                                                        test_size=0.75,
                                                        random_state=123)

    # 1. Init. base estimator to be used
    knn = KNeighborsClassifier()

    # 2. Init. semi-supervised classifier
    ssc = SemiSupervisedClassifier(knn)

    # 3. Supervised classifier training
    ssc = ssc.fit(x_lab, y_lab, _refit=True)

    # 4. Semi-supervised classifier training
    ssc = ssc.fit(x_unlab, _refit=False)

    # 5. Predict
    predictions = ssc.predict(x_test)

    # 6. Accuracy
    accuracy = np.mean(accuracy_score(y_test, predictions))
    print("Accuracy: " + str(accuracy))



