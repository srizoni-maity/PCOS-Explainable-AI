"""
=========================================================
Subset Evaluator

Evaluates Top5 / Top8 / Top10 / Top12 / All

using

LR
RF
XGB
CAT

Returns a dataframe only.

=========================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .benchmark import Benchmark
from src.models.factory import get_models


class SubsetEvaluator:

    def __init__(self):

        self.engine = Benchmark()

        self.subsets = {

            "Top5": "top5.csv",
            "Top8": "top8.csv",
            "Top10": "top10.csv",
            "Top12": "top12.csv",
            "All": "all.csv"

        }

    # --------------------------------------------------

    def evaluate(

        self,

        feature_set,

        X,

        y

    ):

        summary = []

        detailed = []

        for subset_name, filename in self.subsets.items():

            features = pd.read_csv(

                Path(
                    f"outputs/feature_selection/{feature_set}/{filename}"
                )

            )["Feature"].tolist()

            X_subset = X[features]

            results = self.engine.evaluate(

                get_models(),

                X_subset,

                y

            )

            results["Subset"] = subset_name

            detailed.append(results)

            summary.append({

                "Subset": subset_name,

                "NumFeatures": len(features),

                "MeanROC": results["ROC_AUC"].mean(),

                "MeanPR": results["PR_AUC"].mean(),

                "MeanMCC": results["MCC"].mean(),

                "MeanF1": results["F1"].mean(),

                "MeanBrier": results["Brier"].mean()

            })

        summary = pd.DataFrame(summary)

        detailed = pd.concat(

            detailed,

            ignore_index=True

        )

        return summary, detailed