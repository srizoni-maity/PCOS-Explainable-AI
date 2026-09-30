"""
=========================================================
STEP 17

Decision Curve Analysis

IEEE Version

=========================================================
"""

from pathlib import Path
import pickle

import pandas as pd

from src.evaluation.decision_curve import DecisionCurveAnalysis


FEATURE_SET = "HYBRID"


# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("STEP 17 : DECISION CURVE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------

    print("\nLoading model...")

    with open(

        "outputs/models/best_model.pkl",

        "rb"

    ) as f:

        model = pickle.load(f)

    # --------------------------------------------------------

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

    # --------------------------------------------------------

    print("Predicting probabilities...")

    y_prob = model.predict_proba(

        X_test

    )[:, 1]

    # --------------------------------------------------------

    engine = DecisionCurveAnalysis()

    curve = engine.evaluate(

        y_test,

        y_prob

    )

    summary = engine.summary(

        curve

    )

    # --------------------------------------------------------

    output_folder = Path(

        "outputs/decision_curve"

    )

    output_folder.mkdir(

        parents=True,

        exist_ok=True

    )

    curve.to_csv(

        output_folder /

        "decision_curve.csv",

        index=False

    )

    summary.to_csv(

        output_folder /

        "net_benefit_summary.csv",

        index=False

    )

    engine.plot(

        curve,

        output_folder /

        "decision_curve.png"

    )

    # --------------------------------------------------------

    print()

    print("=" * 70)

    print("DECISION CURVE SUMMARY")

    print("=" * 70)

    print(summary)

    print()

    print("Saved to")

    print(output_folder)


# ------------------------------------------------------------

if __name__ == "__main__":

    main()