"""
=========================================================
Publication Figure 6

Permutation Feature Importance

Creates

Figure6_Permutation

=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .base import BaseFigure


class PermutationFigure(BaseFigure):

    def __init__(self):

        super().__init__()

    # ---------------------------------------------------------

    def load_importance(self):

        file = Path(

            "outputs/explainability/permutation_importance.csv"

        )

        return pd.read_csv(file)

    # ---------------------------------------------------------

    def generate(self):

        print("Generating Permutation Importance...")

        importance = self.load_importance()

        importance = importance.sort_values(

            "Importance",

            ascending=True

        )

        top = importance.tail(20)

        self.create()

        plt.barh(

            top["Feature"],

            top["Importance"],

            xerr=top["Std"],

            capsize=3

        )

        self.finish(

            xlabel="Mean Decrease in ROC-AUC",

            ylabel="Feature",

            title="Permutation Feature Importance",

            legend=False,

            grid=True

        )

        self.save(

            "Figure6_Permutation"

        )


def generate():

    PermutationFigure().generate()