"""
=========================================================
Publication Figure 8

Leave-One-Feature-Group-Out (LOFO)

Creates

Figure8_LOFO

=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .base import BaseFigure


class LOFOFigure(BaseFigure):

    def __init__(self):

        super().__init__()

    # ---------------------------------------------------------

    def load_results(self):

        file = Path(

            "outputs/lofo_ablation/comparison.csv"

        )

        return pd.read_csv(file)

    # ---------------------------------------------------------

    def generate(self):

        print(

            "Generating LOFO Figure..."

        )

        df = self.load_results()

        if "Experiment" not in df.columns:

            raise ValueError(

                "comparison.csv must contain an 'Experiment' column."

            )

        if "ROC_AUC" not in df.columns:

            raise ValueError(

                "comparison.csv must contain a 'ROC_AUC' column."

            )

        df = df.sort_values(

            "ROC_AUC",

            ascending=True

        )

        self.create()

        plt.barh(

            df["Experiment"],

            df["ROC_AUC"]

        )

        for i, value in enumerate(df["ROC_AUC"]):

            plt.text(

                value + 0.003,

                i,

                f"{value:.3f}",

                va="center",

                fontsize=9

            )

        self.finish(

            xlabel="ROC-AUC",

            ylabel="LOFO Experiment",

            title="Leave-One-Feature-Group-Out Analysis",

            legend=False,

            grid=True

        )

        self.save(

            "Figure8_LOFO"

        )


def generate():

    LOFOFigure().generate()