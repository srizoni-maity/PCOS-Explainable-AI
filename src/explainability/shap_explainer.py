"""
=========================================================
SHAP Explainer

Computes SHAP values for trained tree-based models.

Supports:
- CatBoost
- XGBoost
- Random Forest

=========================================================
"""

from pathlib import Path
import numpy as np
import shap
import joblib


class SHAPExplainer:

    def __init__(self):

        self.explainer = None

    # -----------------------------------------------------

    def load_model(self, model_path):

        model = joblib.load(model_path)

        self.explainer = shap.TreeExplainer(model)

        return model

    # -----------------------------------------------------

    def explain(self, X):

        shap_values = self.explainer.shap_values(X)

        expected_value = self.explainer.expected_value

        return shap_values, expected_value

    # -----------------------------------------------------

    def save(

        self,

        output_folder,

        shap_values,

        expected_value

    ):

        output_folder = Path(output_folder)

        output_folder.mkdir(

            parents=True,

            exist_ok=True

        )

        np.save(

            output_folder / "shap_values.npy",

            shap_values

        )

        np.save(

            output_folder / "expected_value.npy",

            expected_value

        )