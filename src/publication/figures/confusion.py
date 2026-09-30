"""
=========================================================
Publication Figure 3

Confusion Matrix

=========================================================
"""

import pickle

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from .base import BaseFigure


FEATURE_SET = "HYBRID"


class ConfusionFigure(BaseFigure):

    def __init__(self):

        super().__init__()

    # -----------------------------------------------------

    def generate(self):

        self.create()

        with open(

            "outputs/models/best_model.pkl",

            "rb"

        ) as f:

            model = pickle.load(f)

        X = pd.read_csv(

            f"data/processed/encoded/test_{FEATURE_SET.lower()}.csv"

        )

        y = pd.read_csv(

            "data/processed/test_target.csv"

        ).squeeze()

        y = y.map({

            "No": 0,

            "Yes": 1

        }).astype(int)

        prediction = model.predict(

            X

        )

        cm = confusion_matrix(

            y,

            prediction

        )

        display = ConfusionMatrixDisplay(

            confusion_matrix=cm,

            display_labels=[

                "No PCOS",

                "PCOS"

            ]

        )

        display.plot(

            cmap="Blues",

            colorbar=False

        )

        plt.title(

            "Confusion Matrix"

        )

        plt.tight_layout()

        self.save(

            "Figure3_Confusion"

        )


def generate():

    ConfusionFigure().generate()