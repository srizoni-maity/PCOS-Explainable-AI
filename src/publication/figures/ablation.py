"""
=========================================================
Publication Figure 7

Ablation Study

Creates

Figure7_Ablation

=========================================================
"""

import matplotlib.pyplot as plt
import pandas as pd

from .base import BaseFigure


class AblationFigure(BaseFigure):

    def __init__(self):

        super().__init__()

    # ---------------------------------------------------------

    def load_results(self):

        return pd.read_csv(

            "outputs/ablation/comparison.csv"

        )

    # ---------------------------------------------------------

    def generate(self):

        print()

        print("Generating Ablation Figure...")

        df = self.load_results()

        df = df.sort_values(

            "ROC_AUC",

            ascending=True

        )

        self.create()

        fig, ax = plt.subplots(

            figsize=(8, 4.8)

        )

        bars = ax.barh(

            df["Model"],

            df["ROC_AUC"],

            height=0.60

        )

        ax.set_xlim(

            0.60,

            0.92

        )

        ax.set_xlabel(

            "ROC-AUC"

        )

        ax.set_ylabel(

            "Feature Group"

        )

        ax.set_title(

            "Ablation Study"

        )

        ax.grid(

            axis="x",

            linestyle="--",

            alpha=0.4

        )

        for bar, value in zip(

            bars,

            df["ROC_AUC"]

        ):

            ax.text(

                value + 0.004,

                bar.get_y() + bar.get_height() / 2,

                f"{value:.3f}",

                va="center",

                fontsize=9

            )

        plt.tight_layout()

        self.save(

            "Figure7_Ablation"

        )


# ---------------------------------------------------------


def generate():

    AblationFigure().generate()