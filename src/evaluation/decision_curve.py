"""
=========================================================
Decision Curve Analysis (DCA)

Clinical Utility Evaluation

IEEE / Medical AI Version

=========================================================
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class DecisionCurveAnalysis:

    """
    Decision Curve Analysis.

    Computes

    - Treat None
    - Treat All
    - Model

    Net Benefit

    over a range of threshold probabilities.
    """

    def __init__(

        self,

        thresholds=None

    ):

        if thresholds is None:

            thresholds = np.arange(

                0.05,

                0.60,

                0.01

            )

        self.thresholds = thresholds

    # --------------------------------------------------

    def _net_benefit(

        self,

        y_true,

        y_prob,

        threshold

    ):

        prediction = (

            y_prob >= threshold

        ).astype(int)

        tp = np.sum(

            (prediction == 1)

            &

            (y_true == 1)

        )

        fp = np.sum(

            (prediction == 1)

            &

            (y_true == 0)

        )

        n = len(y_true)

        weight = threshold / (

            1 - threshold

        )

        return (

            tp / n

        ) - (

            fp / n

        ) * weight

    # --------------------------------------------------

    def _treat_all(

        self,

        y_true,

        threshold

    ):

        prevalence = np.mean(

            y_true

        )

        weight = threshold / (

            1 - threshold

        )

        return prevalence - (

            1 - prevalence

        ) * weight

    # --------------------------------------------------

    def evaluate(

        self,

        y_true,

        y_prob

    ):

        rows = []

        for threshold in self.thresholds:

            rows.append({

                "Threshold": threshold,

                "Model":

                    self._net_benefit(

                        y_true,

                        y_prob,

                        threshold

                    ),

                "TreatAll":

                    self._treat_all(

                        y_true,

                        threshold

                    ),

                "TreatNone": 0

            })

        return pd.DataFrame(

            rows

        )

    # --------------------------------------------------

    def summary(

        self,

        df

    ):

        return pd.DataFrame({

            "Strategy":[

                "Model",

                "Treat All",

                "Treat None"

            ],

            "MeanNetBenefit":[

                df["Model"].mean(),

                df["TreatAll"].mean(),

                df["TreatNone"].mean()

            ],

            "MaximumNetBenefit":[

                df["Model"].max(),

                df["TreatAll"].max(),

                df["TreatNone"].max()

            ]

        })

    # --------------------------------------------------

    def plot(

        self,

        df,

        output

    ):

        plt.figure(

            figsize=(8,6)

        )

        plt.plot(

            df["Threshold"],

            df["Model"],

            linewidth=2,

            label="CatBoost"

        )

        plt.plot(

            df["Threshold"],

            df["TreatAll"],

            "--",

            linewidth=2,

            label="Treat All"

        )

        plt.plot(

            df["Threshold"],

            df["TreatNone"],

            "-.",

            linewidth=2,

            label="Treat None"

        )

        plt.xlabel(

            "Threshold Probability"

        )

        plt.ylabel(

            "Net Benefit"

        )

        plt.title(

            "Decision Curve Analysis"

        )

        plt.grid(

            alpha=0.3

        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            output,

            dpi=600

        )

        plt.close()