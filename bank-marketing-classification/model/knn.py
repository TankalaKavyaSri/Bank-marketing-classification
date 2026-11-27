"""K-Nearest Neighbors - votes among the k closest training clients."""
from sklearn.neighbors import KNeighborsClassifier

NAME = "K-Nearest Neighbors"


def build_model(random_state=None):
    return KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
