"""Random Forest - ensemble of 200 decorrelated decision trees."""
from sklearn.ensemble import RandomForestClassifier

NAME = "Random Forest"


def build_model(random_state=42):
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                  random_state=random_state, n_jobs=-1)
