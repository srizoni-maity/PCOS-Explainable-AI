from sklearn.ensemble import RandomForestClassifier

from .base import BaseSelector


class RandomForestSelector(BaseSelector):

    def __init__(self):

        self.model = RandomForestClassifier(

            n_estimators=500,

            random_state=42,

            n_jobs=-1

        )

    def fit(self, X, y):

        self.model.fit(

            X,

            y

        )

        scores = self.model.feature_importances_

        return self.normalize(scores)