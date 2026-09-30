import numpy as np

from sklearn.linear_model import LogisticRegression

from .base import BaseSelector


class LassoSelector(BaseSelector):

    def __init__(self):

        self.model = LogisticRegression(

            penalty="l1",

            solver="liblinear",

            C=1.0,

            max_iter=5000,

            random_state=42

        )

    def fit(self, X, y):

        self.model.fit(

            X,

            y

        )

        scores = np.abs(

            self.model.coef_[0]

        )

        return self.normalize(scores)