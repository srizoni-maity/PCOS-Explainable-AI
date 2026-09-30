"""
=========================================================
Publication Figure 10

Decision Curve Analysis

Creates

Figure10_DCA

=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .base import BaseFigure


class DecisionCurveFigure(BaseFigure):

    def __init__(self):

        super().__init__()

    # ---------------------------------------------------------

    def load_curve(self):

        file = Path(

            "outputs/decision_curve/decision_curve.csv"

        )

        return pd.read_csv(file)

    # ---------------------------------------------------------

    def load_summary(self):

        file = Path(

            "outputs/decision_curve/net_benefit_summary.csv"

        )

        return pd.read_csv(file)

    # ---------------------------------------------------------

    def generate(self):

        print(

            "Generating Decision Curve Figure..."

        )

        df = self.load_curve()

        summary = self.load_summary()

        threshold_col = df.columns[0]

        threshold = df[threshold_col]

        model_col = None
        treat_all_col = None
        treat_none_col = None

        for col in df.columns:

            lower = col.lower()

            if "model" in lower:

                model_col = col

            elif "all" in lower:

                treat_all_col = col

            elif "none" in lower:

                treat_none_col = col

        if model_col is None:

            raise ValueError(

                "Model column not found in decision_curve.csv"

            )

        self.create()

        plt.plot(

            threshold,

            df[model_col],

            linewidth=2.5,

            label="CatBoost"

        )

        if treat_all_col is not None:

            plt.plot(

                threshold,

                df[treat_all_col],

                "--",

                linewidth=1.5,

                label="Treat All"

            )

        if treat_none_col is not None:

            plt.plot(

                threshold,

                df[treat_none_col],

                "-.",

                linewidth=1.5,

                label="Treat None"

            )

        if {

            "Strategy",

            "MeanNetBenefit"

        }.issubset(summary.columns):

            model_row = summary[

                summary["Strategy"].str.lower() == "model"

            ]

            if not model_row.empty:

                mean_nb = float(

                    model_row["MeanNetBenefit"].iloc[0]

                )

                plt.text(

                    0.60,

                    0.05,

                    f"Mean Net Benefit = {mean_nb:.3f}",

                    fontsize=10,

                    bbox=dict(

                        facecolor="white",

                        alpha=0.8,

                        edgecolor="gray"

                    )

                )

        plt.xlim(0, 1)

        self.finish(

            xlabel="Threshold Probability",

            ylabel="Net Benefit",

            title="Decision Curve Analysis",

            legend=True,

            grid=True

        )

        self.save(

            "Figure10_DCA"

        )


def generate():

    DecisionCurveFigure().generate()