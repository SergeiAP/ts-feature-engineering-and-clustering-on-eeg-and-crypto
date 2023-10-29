import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def run_baseline(X_train: np.ndarray | pd.DataFrame,
                 y_train: np.ndarray | pd.Series,
                 X_test: np.ndarray | pd.DataFrame,
                 y_test: np.ndarray | pd.Series,
                 seed: int,
                 n_estimators: int = 100) -> RandomForestClassifier:
    """_summary_

    Args:
        X_train (np.ndarray | pd.DataFrame): _description_
        y_train (np.ndarray | pd.Series): _description_
        X_test (np.ndarray | pd.DataFrame): _description_
        y_test (np.ndarray | pd.Series): _description_
        seed (int): _description_
        n_estimators (int, optional): _description_. Defaults to 100.

    Returns:
        RandomForestClassifier: _description_
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_train, y_train)
    print("Accuracy on training set is : {:.2f}%".format(
        100 * clf.score(X_train, y_train))
        )
    print("Accuracy on test set is : {:.2f}%".format(
        100 * clf.score(X_test, y_test))
        )
    Y_test_pred = clf.predict(X_test)
    print(classification_report(y_test, Y_test_pred))
    return clf
