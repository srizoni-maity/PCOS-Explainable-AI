"""
=========================================================
STEP 18

Bootstrap Confidence Intervals

IEEE Publication Version

=========================================================
"""

from pathlib import Path
import pickle

import pandas as pd

from src.evaluation.confidence_interval import BootstrapEvaluator


FEATURE_SET = "HYBRID"


def main():

    print("=" * 70)
    print("STEP 18 : BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    # -----------------------------------------------------

    print("\nLoading model...")

    with open(
        "outputs/models/best_model.pkl",
        "rb"
    ) as f:

        model = pickle.load(f)

    # -----------------------------------------------------

    print("Loading test dataset...")

    X_test = pd.read_csv(
        f"data/processed/encoded/test_{FEATURE_SET.lower()}.csv"
    )

    y_test = pd.read_csv(
        "data/processed/test_target.csv"
    ).squeeze()

    LABEL_MAP = {

        "No": 0,
        "Yes": 1

    }

    y_test = y_test.map(
        LABEL_MAP
    ).astype(int)

    # -----------------------------------------------------

    output_folder = Path(
        "outputs/confidence_intervals"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nRunning Bootstrap...")

    evaluator = BootstrapEvaluator(

        n_bootstrap=2000,

        confidence=0.95,

        random_state=42

    )

    samples = evaluator.bootstrap(

        model,

        X_test,

        y_test

    )

    summary = evaluator.confidence_interval(

        samples

    )

    samples.to_csv(

        output_folder /

        "bootstrap_samples.csv",

        index=False

    )

    summary.to_csv(

        output_folder /

        "confidence_intervals.csv",

        index=False

    )

    evaluator.plot_distributions(

        samples,

        output_folder

    )

    evaluator.plot_boxplots(

        samples,

        output_folder /

        "boxplots.png"

    )

    print()

    print("=" * 70)

    print("95% CONFIDENCE INTERVALS")

    print("=" * 70)

    print(

        evaluator.print_summary(

            summary

        )

    )

    print()

    print("Saved to")

    print(output_folder)


if __name__ == "__main__":

    main()