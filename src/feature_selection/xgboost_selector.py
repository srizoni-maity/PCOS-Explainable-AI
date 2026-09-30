import numpy as np

from xgboost import XGBClassifier

from .base import BaseSelector


class XGBoostSelector(BaseSelector):

    def __init__(self):

        self.model = XGBClassifier(

            n_estimators=300,

            max_depth=4,

            learning_rate=0.05,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=42,

            objective="binary:logistic",

            eval_metric="logloss",

            n_jobs=-1,
        )

    def fit(self, X, y):

        self.model.fit(X, y)

        scores = self.model.feature_importances_

        scores = np.nan_to_num(scores)

        return self.normalize(scores)