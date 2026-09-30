"""
=========================================================
Publication Figure 2

Precision Recall Curve

=========================================================
"""

import pickle

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

from .base import BaseFigure


FEATURE_SET = "HYBRID"


class PRFigure(BaseFigure):

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

        probability = model.predict_proba(

            X

        )[:, 1]

        precision, recall, _ = precision_recall_curve(

            y,

            probability

        )

        ap = average_precision_score(

            y,

            probability

        )

        plt.plot(

            recall,

            precision,

            linewidth=2,

            label=f"CatBoost (AP = {ap:.3f})"

        )

        self.finish(

            xlabel="Recall",

            ylabel="Precision",

            title="Precision–Recall Curve",

            legend=True,

            grid=True

        )

        self.save(

            "Figure2_PR"

        )


def generate():

    PRFigure().generate()