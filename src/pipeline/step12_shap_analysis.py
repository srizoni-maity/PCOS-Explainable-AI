"""
=========================================================
STEP 12

SHAP Explainability Analysis

Generates

• SHAP Values
• Summary Beeswarm
• Summary Bar
• Dependence Plots
• Waterfall Plots
• Force Plot
• Feature Importance CSV
• JSON Report

=========================================================
"""

from pathlib import Path
import json

import pandas as pd

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.shap_plots import SHAPPlots
from src.explainability.shap_report import SHAPReport


FEATURE_SET = "HYBRID"


# -----------------------------------------------------


def load_best_subset():

    with open(

        f"outputs/subset_selection/{FEATURE_SET}/best_subset.json",

        "r"

    ) as f:

        info = json.load(f)

    subset = info["selected_subset"]

    filename = subset.lower() + ".csv"

    features = pd.read_csv(

        f"outputs/feature_selection/{FEATURE_SET}/{filename}"

    )["Feature"].tolist()

    return subset, features


# -----------------------------------------------------


def main():

    print("=" * 70)
    print("STEP 12 : SHAP EXPLAINABILITY")
    print("=" * 70)

    subset_name, features = load_best_subset()

    print()
    print(f"Feature Set : {FEATURE_SET}")
    print(f"Subset      : {subset_name}")
    print(f"Features    : {len(features)}")

    # -------------------------------------------------

    X = pd.read_csv(

        f"data/processed/encoded/test_{FEATURE_SET.lower()}.csv"

    )

    X = X[features]

    # -------------------------------------------------

    model_path = "outputs/models/best_model.pkl"

    explainer = SHAPExplainer()

    model = explainer.load_model(

        model_path

    )

    shap_values, expected_value = explainer.explain(

        X

    )

    output_folder = Path(

        "outputs/explainability"

    )

    explainer.save(

        output_folder,

        shap_values,

        expected_value

    )

    # -------------------------------------------------

    plots = SHAPPlots()

    print("\nGenerating SHAP plots...")

    plots.summary_beeswarm(

        shap_values,

        X,

        output_folder

    )

    plots.summary_bar(

        shap_values,

        X,

        output_folder

    )

    plots.dependence_plots(

        shap_values,

        X,

        output_folder,

        top_k=10

    )

    plots.waterfall_plots(

        explainer.explainer,

        shap_values,

        X,

        output_folder,

        n_patients=5

    )

    plots.force_plot(

        explainer.explainer,

        shap_values,

        X,

        output_folder,

        patient=0

    )

    # -------------------------------------------------

    report = SHAPReport()

    importance = report.generate(

        shap_values,

        X.columns,

        "CatBoost",

        len(X),

        output_folder

    )

    # -------------------------------------------------

    print()

    print("=" * 70)
    print("TOP 10 FEATURES")
    print("=" * 70)

    print(

        importance.head(10)

    )

    print()

    print("Saved to")

    print(output_folder)


# -----------------------------------------------------

if __name__ == "__main__":

    main()