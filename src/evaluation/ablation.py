"""
=========================================================
Ablation Study Engine

IEEE Version

Compares

1. Full Model
2. Clinical Only
3. Lifestyle Only

=========================================================
"""

from sklearn.base import clone

import pandas as pd

from src.training.metrics import calculate_metrics


class AblationStudy:

    def __init__(self):

        pass

    # ---------------------------------------------------------

    def feature_groups(self, X):

        symptom_features = [

            c

            for c in X.columns

            if c.startswith("Symptom_")

        ]

        if "SymptomCount" in X.columns:

            symptom_features.insert(

                0,

                "SymptomCount"

            )

        clinical = symptom_features

        lifestyle = [

            c

            for c in X.columns

            if c not in clinical

        ]

        return {

            "Clinical": clinical,

            "Lifestyle": lifestyle,

            "Full": list(X.columns)

        }

    # ---------------------------------------------------------

    def evaluate(

        self,

        model_definition,

        X_train,

        y_train,

        X_test,

        y_test

    ):

        groups = self.feature_groups(

            X_train

        )

        comparison = []

        detailed = {}

        for name, features in groups.items():

            print()

            print("=" * 60)

            print(name)

            print("=" * 60)

            print(

                f"Features : {len(features)}"

            )

            model = clone(

                model_definition.estimator()

            )

            model.fit(

                X_train[features],

                y_train

            )

            pred = model.predict(

                X_test[features]

            )

            prob = model.predict_proba(

                X_test[features]

            )[:, 1]

            metrics = calculate_metrics(

                y_test,

                pred,

                prob

            )

            row = {

                "Model": name,

                "NumFeatures": len(features)

            }

            row.update(metrics)

            comparison.append(row)

            detailed[name] = pd.DataFrame(

                [

                    {

                        "Metric": k,

                        "Value": v

                    }

                    for k, v in metrics.items()

                ]

            )

        comparison = pd.DataFrame(

            comparison

        )

        comparison = comparison.sort_values(

            "ROC_AUC",

            ascending=False

        )

        return {

            "comparison": comparison,

            "details": detailed,

            "groups": groups

        }