from xgboost import XGBClassifier

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    
    @property
    def name(self):
        return "XGBoost"

    def estimator(self):

        return XGBClassifier(

            random_state=42,

            objective="binary:logistic",

            eval_metric="logloss",

            n_jobs=-1

        )

    def search_space(self, trial):

        return {

            "n_estimators":

            trial.suggest_int(

                "n_estimators",

                100,

                800

            ),

            "max_depth":

            trial.suggest_int(

                "max_depth",

                3,

                10

            ),

            "learning_rate":

            trial.suggest_float(

                "learning_rate",

                0.01,

                0.3,

                log=True

            ),

            "subsample":

            trial.suggest_float(

                "subsample",

                0.6,

                1.0

            ),

            "colsample_bytree":

            trial.suggest_float(

                "colsample_bytree",

                0.6,

                1.0

            )

        }