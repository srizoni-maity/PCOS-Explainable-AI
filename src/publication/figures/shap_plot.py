"""
=========================================================
Publication Figure 4 & 5

SHAP Explainability Figures

Creates

Figure4_SHAP_Beeswarm
Figure5_SHAP_Bar

=========================================================
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import shap

from .base import BaseFigure


FEATURE_SET = "HYBRID"


class SHAPFigure(BaseFigure):

    def __init__(self):

        super().__init__()

    # ---------------------------------------------------------

    def load_model(self):

        with open(

            "outputs/models/best_model.pkl",

            "rb"

        ) as f:

            return pickle.load(f)

    # ---------------------------------------------------------

    def load_dataset(self):

        X = pd.read_csv(

            f"data/processed/encoded/test_{FEATURE_SET.lower()}.csv"

        )

        return X

    # ---------------------------------------------------------

    def compute_shap(self):

        model = self.load_model()

        X = self.load_dataset()

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):

            shap_values = shap_values[1]

        return X, shap_values

    # ---------------------------------------------------------

    def beeswarm(self):

        X, shap_values = self.compute_shap()

        self.create()

        shap.summary_plot(

            shap_values,

            X,

            show=False

        )

        self.finish(

            xlabel="SHAP Value",

            ylabel="Feature",

            title="SHAP Summary (Beeswarm)",

            legend=False,

            grid=False

        )

        self.save(

            "Figure4_SHAP_Beeswarm"

        )

    # ---------------------------------------------------------

    def bar(self):

        X, shap_values = self.compute_shap()

        self.create()

        shap.summary_plot(

            shap_values,

            X,

            plot_type="bar",

            show=False

        )

        self.finish(

            xlabel="Mean |SHAP Value|",

            ylabel="Feature",

            title="SHAP Feature Importance",

            legend=False,

            grid=False

        )

        self.save(

            "Figure5_SHAP_Bar"

        )

    # ---------------------------------------------------------

    def generate(self):

        print("Generating SHAP Beeswarm...")

        self.beeswarm()

        print("Generating SHAP Bar...")

        self.bar()


def generate():

    SHAPFigure().generate()