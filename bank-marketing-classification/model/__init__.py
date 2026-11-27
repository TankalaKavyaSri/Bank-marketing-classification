"""Model package: one .py file per classifier, plus a helper that returns all five."""
from . import logistic_regression, decision_tree, knn, naive_bayes, random_forest

_MODULES = [logistic_regression, decision_tree, knn, naive_bayes, random_forest]


def get_models(random_state=42):
    """Return {display name: fresh, untrained estimator} for all five models."""
    return {m.NAME: m.build_model(random_state=random_state) for m in _MODULES}
