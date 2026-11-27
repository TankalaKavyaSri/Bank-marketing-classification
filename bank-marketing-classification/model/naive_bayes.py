"""Gaussian Naive Bayes - Bayes rule with a per-feature bell curve per class."""
from sklearn.naive_bayes import GaussianNB

NAME = "Naive Bayes (Gaussian)"


def build_model(random_state=None):
    return GaussianNB()
