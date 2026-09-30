"""
=========================================================
Calibration Evaluation

IEEE Medical AI Version

=========================================================
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class CalibrationEvaluator:

    def __init__(

        self,

        n_bins=10

    ):

        self.n_bins = n_bins

    # ----------------------------------------------------

    def evaluate(

        self,

        y_true,

        y_prob

    ):

        frac_pos, mean_pred = calibration_curve(

            y_true,

            y_prob,

            n_bins=self.n_bins,

            strategy="uniform"

        )

        brier = brier_score_loss(

            y_true,

            y_prob

        )

        ece = self.expected_calibration_error(

            y_true,

            y_prob

        )

        curve = pd.DataFrame({

            "MeanPredictedProbability": mean_pred,

            "ObservedFrequency": frac_pos

        })

        metrics = pd.DataFrame({

            "Metric": [

                "Brier",

                "ECE"

            ],

            "Value": [

                brier,

                ece

            ]

        })

        return curve, metrics

    # ----------------------------------------------------

    def expected_calibration_error(

        self,

        y_true,

        y_prob

    ):

        bins = np.linspace(

            0,

            1,

            self.n_bins + 1

        )

        ece = 0

        total = len(y_true)

        for i in range(self.n_bins):

            mask = (

                (y_prob >= bins[i])

                &

                (y_prob < bins[i+1])

            )

            if np.sum(mask) == 0:

                continue

            acc = np.mean(

                y_true[mask]

            )

            conf = np.mean(

                y_prob[mask]

            )

            ece += (

                np.sum(mask)

                /

                total

            ) * abs(

                acc - conf

            )

        return ece

    # ----------------------------------------------------

    def plot_curve(

        self,

        curve,

        output

    ):

        plt.figure(

            figsize=(6,6)

        )

        plt.plot(

            [0,1],

            [0,1],

            "--"

        )

        plt.plot(

            curve["MeanPredictedProbability"],

            curve["ObservedFrequency"],

            marker="o",

            linewidth=2

        )

        plt.xlabel(

            "Predicted Probability"

        )

        plt.ylabel(

            "Observed Frequency"

        )

        plt.title(

            "Reliability Diagram"

        )

        plt.tight_layout()

        plt.savefig(

            output,

            dpi=600

        )

        plt.close()

    # ----------------------------------------------------

    def probability_histogram(

        self,

        y_prob,

        output

    ):

        plt.figure(

            figsize=(7,4)

        )

        plt.hist(

            y_prob,

            bins=20

        )

        plt.xlabel(

            "Predicted Probability"

        )

        plt.ylabel(

            "Count"

        )

        plt.tight_layout()

        plt.savefig(

            output,

            dpi=600

        )

        plt.close()