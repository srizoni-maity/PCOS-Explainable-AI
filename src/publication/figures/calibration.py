"""
=========================================================
Publication Figure 9

Calibration Curve

Creates

Figure9_Calibration

=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .base import BaseFigure


class CalibrationFigure(BaseFigure):

    def __init__(self):

        super().__init__()

    # ---------------------------------------------------------

    def load_curve(self):

        file = Path(

            "outputs/calibration/calibration_curve.csv"

        )

        return pd.read_csv(file)

    # ---------------------------------------------------------

    def load_metrics(self):

        file = Path(

            "outputs/calibration/calibration_metrics.csv"

        )

        return pd.read_csv(file)

    # ---------------------------------------------------------

    def generate(self):

        print(

            "Generating Calibration Figure..."

        )

        curve = self.load_curve()

        metrics = self.load_metrics()

        # ---------------------------------------------

        x = curve.iloc[:, 0]

        y = curve.iloc[:, 1]

        brier = float(

            metrics.loc[

                metrics["Metric"] == "Brier",

                "Value"

            ].values[0]

        )

        ece = float(

            metrics.loc[

                metrics["Metric"] == "ECE",

                "Value"

            ].values[0]

        )

        self.create()

        plt.plot(

            x,

            y,

            linewidth=2.5,

            marker="o",

            label=f"CatBoost (Brier={brier:.3f})"

        )

        plt.plot(

            [0, 1],

            [0, 1],

            "--",

            linewidth=1.5,

            label="Perfect Calibration"

        )

        plt.text(

            0.60,

            0.12,

            f"ECE = {ece:.3f}",

            fontsize=10,

            bbox=dict(

                facecolor="white",

                alpha=0.8,

                edgecolor="gray"

            )

        )

        plt.xlim(0, 1)

        plt.ylim(0, 1)

        self.finish(

            xlabel="Mean Predicted Probability",

            ylabel="Observed Frequency",

            title="Calibration Curve",

            legend=True,

            grid=True

        )

        self.save(

            "Figure9_Calibration"

        )


def generate():

    CalibrationFigure().generate()