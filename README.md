# Semi-Supervised Classifier (SSC)
Enables semi-supervised learning for sci-kit classifiers

## Requirements
* Python 3.6.
* NumPy 1.18.5+
* scikit-learn

### Dependencies in order for the project to run are:
`numpy, sci-kit learn and pandas`

## Usage
Import SemiSuperviesdClassifier class into your working file SemiSuperviesdClassifier class then you can initialize it and use in your code. As shown in the example bellow.

### Example

```python
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from <path> import SemiSupervisedClassifier

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
```

### Output
Output of the classifier depend on the method called, it can return supervised or semi-supervised trained classifier if called fit(), and it returns predictions for (test) unlabeled data set.

## Datasets
Used datasets are taken from the [Kaggle](https://www.kaggle.com/) website, a largely used platform by machine learning enthusiasts.

* [Income evaluation](https://www.kaggle.com/lodetomasi1995/income-classification)
* [Mobile price range](https://www.kaggle.com/iabhishekofficial/mobile-price-classification) 
* [Meal demand forecasting](https://www.kaggle.com/sureshmecad/meal-demand-forecasting)

#### Disclaimer
_The goal of the project is to create sci-kit learn compatible classifier enabling semi-supervised learning. And with it to simplify the procces of labeling data._
