"""
=========================================================
Evaluation Plots
=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


class EvaluationPlots:

    def __init__(self):

        self.output = Path(
            "outputs/final_evaluation/figures"
        )

        self.output.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------

    def roc_curve(self, y_true, y_prob):

        plt.figure(figsize=(6, 6))

        RocCurveDisplay.from_predictions(
            y_true,
            y_prob
        )

        plt.title("ROC Curve")

        plt.savefig(
            self.output / "roc_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------

    def pr_curve(self, y_true, y_prob):

        plt.figure(figsize=(6, 6))

        PrecisionRecallDisplay.from_predictions(
            y_true,
            y_prob
        )

        plt.title("Precision–Recall Curve")

        plt.savefig(
            self.output / "pr_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------

    def calibration_curve(self, y_true, y_prob):

        prob_true, prob_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=10
        )

        plt.figure(figsize=(6, 6))

        plt.plot(prob_pred, prob_true, marker="o")

        plt.plot([0, 1], [0, 1], "--")

        plt.xlabel("Predicted Probability")

        plt.ylabel("Observed Frequency")

        plt.title("Calibration Curve")

        plt.savefig(
            self.output / "calibration_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------

    def confusion_matrix(self, y_true, y_pred):

        plt.figure(figsize=(5, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred
        )

        plt.title("Confusion Matrix")

        plt.savefig(
            self.output / "confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()