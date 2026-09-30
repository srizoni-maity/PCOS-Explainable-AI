from pathlib import Path
import json
import joblib
import pandas as pd


class FinalTrainer:

    def __init__(self):

        self.output = Path("outputs/models")

        self.output.mkdir(
            parents=True,
            exist_ok=True
        )

    def train(

        self,

        model_definition,

        params,

        X,

        y

    ):

        model = model_definition.estimator()

        model.set_params(

            **params

        )

        model.fit(

            X,

            y

        )

        joblib.dump(

            model,

            self.output / "best_model.pkl"

        )

        with open(

            self.output / "best_params.json",

            "w"

        ) as f:

            json.dump(

                params,

                f,

                indent=4

            )

        info = {

            "Model": model_definition.name,

            "TrainingSamples": len(X),

            "Features": list(X.columns)

        }

        with open(

            self.output / "best_model_info.json",

            "w"

        ) as f:

            json.dump(

                info,

                f,

                indent=4

            )

        return model