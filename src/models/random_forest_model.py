from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    
    @property
    def name(self):
        return "Random Forest"

    def estimator(self):

        return RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

    def search_space(self, trial):

        return {

            "n_estimators":

            trial.suggest_int(

                "n_estimators",

                200,

                800

            ),

            "max_depth":

            trial.suggest_int(

                "max_depth",

                3,

                20

            ),

            "min_samples_split":

            trial.suggest_int(

                "min_samples_split",

                2,

                20

            ),

            "min_samples_leaf":

            trial.suggest_int(

                "min_samples_leaf",

                1,

                10

            ),

            "max_features":

            trial.suggest_categorical(

                "max_features",

                [

                    "sqrt",

                    "log2",

                    None

                ]

            )

        }