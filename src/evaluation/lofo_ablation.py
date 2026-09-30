"""
=========================================================
Leave-One-Feature-Group-Out Ablation Study

IEEE Version

Experiments

1. Full
2. No Symptoms
3. No Diet
4. No Lifestyle
5. No Awareness
6. No Medical History

=========================================================
"""

from sklearn.base import clone

import pandas as pd

from src.training.metrics import calculate_metrics


class LOFOAblation:

    """
    Leave-One-Feature-Group-Out Ablation Study.
    """

    def __init__(self):

        pass

    # ---------------------------------------------------------

    def feature_groups(self):

        return {

            "Symptoms": [

                "SymptomCount",

                "Symptom_IrregularCycle",

                "Symptom_Acne",

                "Symptom_Hirsutism",

                "Symptom_WeightGain",

                "Symptom_None"

            ],

            "Diet": [

                "PackagedFood",

                "PackagedFoodScore",

                "FastFood",

                "FastFoodScore",

                "FoodLabel",

                "FoodLabelScore",

                "DietRiskScore",

                "TransFatAwareness"

            ],

            "Lifestyle": [

                "Exercise",

                "ExerciseScore",

                "Sleep",

                "SleepScore",

                "Stress",

                "StressScore",

                "SugaryDrinks",

                "LifestyleRiskScore"

            ],

            "Awareness": [

                "PCOSAwareness",

                "PCOSAwarenessScore"

            ],

            "MedicalHistory": [

                "MedicalHistory",

                "MedicalRiskScore"

            ]

        }

    # ---------------------------------------------------------

    def build_experiments(

        self,

        X

    ):

        groups = self.feature_groups()

        experiments = {

            "Full": list(X.columns)

        }

        for group_name, remove_features in groups.items():

            experiments[f"No{group_name}"] = [

                feature

                for feature in X.columns

                if feature not in remove_features

            ]

        return experiments

    # ---------------------------------------------------------

    def evaluate(

        self,

        model_definition,

        X_train,

        y_train,

        X_test,

        y_test

    ):

        experiments = self.build_experiments(

            X_train

        )

        rows = []

        details = {}

        for experiment, features in experiments.items():

            print()

            print("=" * 60)

            print(experiment)

            print("=" * 60)

            print(

                f"Features : {len(features)}"

            )

            estimator = clone(

                model_definition.estimator()

            )

            estimator.fit(

                X_train[features],

                y_train

            )

            pred = estimator.predict(

                X_test[features]

            )

            prob = estimator.predict_proba(

                X_test[features]

            )[:, 1]

            metrics = calculate_metrics(

                y_test,

                pred,

                prob

            )

            row = {

                "Experiment": experiment,

                "NumFeatures": len(features)

            }

            row.update(metrics)

            rows.append(row)

            details[experiment] = pd.DataFrame(

                [

                    {

                        "Metric": k,

                        "Value": v

                    }

                    for k, v in metrics.items()

                ]

            )

        comparison = pd.DataFrame(

            rows

        )

        comparison = comparison.sort_values(

            "ROC_AUC",

            ascending=False

        )

        return {

            "comparison": comparison,

            "details": details,

            "groups": self.feature_groups()

        }

    # ---------------------------------------------------------

    def performance_drop(

        self,

        comparison

    ):

        full = comparison.loc[

            comparison["Experiment"] == "Full"

        ].iloc[0]

        rows = []

        for _, row in comparison.iterrows():

            rows.append({

                "Experiment": row["Experiment"],

                "DeltaROC_AUC": row["ROC_AUC"] - full["ROC_AUC"],

                "DeltaPR_AUC": row["PR_AUC"] - full["PR_AUC"],

                "DeltaF1": row["F1"] - full["F1"],

                "DeltaMCC": row["MCC"] - full["MCC"],

                "DeltaAccuracy": row["Accuracy"] - full["Accuracy"]

            })

        return pd.DataFrame(rows)