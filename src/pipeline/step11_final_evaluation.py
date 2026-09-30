"""
=========================================================
STEP 11

Final Test Evaluation

=========================================================
"""

import json
import pandas as pd

from src.evaluation.final_evaluator import FinalEvaluator
from src.evaluation.plots import EvaluationPlots


FEATURE_SET = "HYBRID"


def load_features():

    with open(
        f"outputs/subset_selection/{FEATURE_SET}/best_subset.json"
    ) as f:

        info = json.load(f)

    subset = info["selected_subset"]

    return pd.read_csv(

        f"outputs/feature_selection/{FEATURE_SET}/{subset.lower()}.csv"

    )["Feature"].tolist()


def main():

    print("=" * 70)
    print("STEP 11 : FINAL TEST EVALUATION")
    print("=" * 70)

    features = load_features()

    X_test = pd.read_csv(

        f"data/processed/encoded/test_{FEATURE_SET.lower()}.csv"

    )[features]

    y_test = pd.read_csv(
        "data/processed/test_target.csv"
    ).squeeze()

    y_test = y_test.map({

        "No": 0,

        "Yes": 1

    }).astype(int)

    evaluator = FinalEvaluator()

    results = evaluator.evaluate(X_test, y_test)

    plots = EvaluationPlots()

    plots.roc_curve(results["y_true"], results["y_prob"])

    plots.pr_curve(results["y_true"], results["y_prob"])

    plots.calibration_curve(
        results["y_true"],
        results["y_prob"]
    )

    plots.confusion_matrix(
        results["y_true"],
        results["y_pred"]
    )

    print()
    print(results["metrics"])

    print()
    print("Saved to outputs/final_evaluation")


if __name__ == "__main__":

    main()