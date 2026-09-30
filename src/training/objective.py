"""
=========================================================
Generic Optuna Objective Function
=========================================================
"""

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from .metrics import calculate_metrics


class Objective:

    """
    Generic Optuna objective for any model.
    """

    def __init__(
        self,
        model_definition,
        X,
        y,
        scoring="ROC_AUC",
        n_splits=5,
        random_state=42,
    ):

        self.model_definition = model_definition
        self.X = X
        self.y = y
        self.scoring = scoring

        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    def __call__(self, trial):

        params = self.model_definition.search_space(trial)

        model = self.model_definition.estimator()

        model.set_params(**params)

        scores = []

        for train_idx, valid_idx in self.cv.split(self.X, self.y):

            X_train = self.X.iloc[train_idx]
            X_valid = self.X.iloc[valid_idx]

            y_train = self.y.iloc[train_idx]
            y_valid = self.y.iloc[valid_idx]

            estimator = clone(model)

            estimator.fit(
                X_train,
                y_train
            )

            y_prob = estimator.predict_proba(
                X_valid
            )[:, 1]

            y_pred = estimator.predict(
                X_valid
            )

            metrics = calculate_metrics(
                y_valid,
                y_pred,
                y_prob
            )

            scores.append(
                metrics[self.scoring]
            )

        return float(np.mean(scores))