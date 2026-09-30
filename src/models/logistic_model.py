from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticModel(BaseModel):
    @property
    def name(self):
        return "Logistic Regression"

    def estimator(self):

        return LogisticRegression(
            random_state=42,
            max_iter=5000
        )

    def search_space(self, trial):

        return {

            "C": trial.suggest_float(
                "C",
                1e-3,
                100,
                log=True
            ),

            "solver": trial.suggest_categorical(
                "solver",
                [
                    "liblinear",
                    "lbfgs"
                ]
            )

        }