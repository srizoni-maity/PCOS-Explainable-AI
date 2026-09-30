from sklearn.feature_selection import mutual_info_classif

from .base import BaseSelector


class MutualInformationSelector(BaseSelector):

    def fit(self, X, y):

        scores = mutual_info_classif(

            X,

            y,

            random_state=42

        )

        return self.normalize(scores)