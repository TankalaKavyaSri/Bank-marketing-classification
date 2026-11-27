"""Decision Tree - a flow-chart of yes/no questions chosen by Gini impurity."""
from sklearn.tree import DecisionTreeClassifier

NAME = "Decision Tree"


def build_model(random_state=42):
    return DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=random_state)
