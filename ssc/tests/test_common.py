import pytest

from sklearn.utils.estimator_checks import check_estimator

from ssc import TemplateEstimator
from ssc import TemplateClassifier
from ssc import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
