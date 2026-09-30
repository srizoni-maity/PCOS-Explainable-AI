from catboost import CatBoostClassifier

from .base_model import BaseModel


class CatBoostModel(BaseModel):

    @property
    def name(self):
        return "CatBoost"

    def estimator(self):

        return CatBoostClassifier(

            verbose=False,

            random_seed=42

        )

    def search_space(self, trial):

        return {

            "iterations":

            trial.suggest_int(

                "iterations",

                100,

                800

            ),

            "depth":

            trial.suggest_int(

                "depth",

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

            "l2_leaf_reg":

            trial.suggest_float(

                "l2_leaf_reg",

                1,

                10

            )

        }