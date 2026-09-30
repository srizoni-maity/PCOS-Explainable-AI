import numpy as np

from sklearn.feature_selection import f_classif

from .base import BaseSelector


class AnovaSelector(BaseSelector):

    def fit(self, X, y):

        scores, _ = f_classif(

            X,

            y

        )

        scores = np.nan_to_num(scores)

        return self.normalize(scores)