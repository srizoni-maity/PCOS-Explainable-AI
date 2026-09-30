"""
=========================================================
STEP 16

Calibration Analysis

=========================================================
"""

from pathlib import Path
import pickle

import pandas as pd

from src.evaluation.calibration import CalibrationEvaluator


FEATURE_SET = "HYBRID"


def main():

    print("="*70)
    print("STEP 16 : CALIBRATION ANALYSIS")
    print("="*70)

    model = pickle.load(

        open(

            "outputs/models/best_model.pkl",

            "rb"

        )

    )

    X_test = pd.read_csv(

        f"data/processed/encoded/test_{FEATURE_SET.lower()}.csv"

    )

    y_test = pd.read_csv(

        "data/processed/test_target.csv"

    ).squeeze()

    y_test = y_test.map({

        "No":0,

        "Yes":1

    }).astype(int)

    y_prob = model.predict_proba(

        X_test

    )[:,1]

    engine = CalibrationEvaluator()

    curve, metrics = engine.evaluate(

        y_test,

        y_prob

    )

    output = Path(

        "outputs/calibration"

    )

    output.mkdir(

        parents=True,

        exist_ok=True

    )

    curve.to_csv(

        output/"calibration_curve.csv",

        index=False

    )

    metrics.to_csv(

        output/"calibration_metrics.csv",

        index=False

    )

    engine.plot_curve(

        curve,

        output/"reliability_diagram.png"

    )

    engine.probability_histogram(

        y_prob,

        output/"probability_histogram.png"

    )

    print()
    print("="*70)
    print("CALIBRATION METRICS")
    print("="*70)
    print(metrics)

    print()
    print("Saved to")
    print(output)


if __name__ == "__main__":

    main()