"""Logistic Regression - linear model that outputs P(yes) through a sigmoid."""
from sklearn.linear_model import LogisticRegression

NAME = "Logistic Regression"


def build_model(random_state=42):
    return LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
