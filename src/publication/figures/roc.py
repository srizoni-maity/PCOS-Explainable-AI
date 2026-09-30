"""
=========================================================
Publication Figure 1

ROC Curve

=========================================================
"""

import pickle

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from .base import BaseFigure


FEATURE_SET = "HYBRID"


class ROCFigure(BaseFigure):

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

        fpr, tpr, _ = roc_curve(

            y,

            probability

        )

        auc_score = auc(

            fpr,

            tpr

        )

        plt.plot(

            fpr,

            tpr,

            linewidth=2,

            label=f"CatBoost (AUC = {auc_score:.3f})"

        )

        plt.plot(

            [0, 1],

            [0, 1],

            "--",

            linewidth=1,

            label="Random"

        )

        self.finish(

            xlabel="False Positive Rate",

            ylabel="True Positive Rate",

            title="Receiver Operating Characteristic",

            legend=True,

            grid=True

        )

        self.save(

            "Figure1_ROC"

        )


def generate():

    ROCFigure().generate()