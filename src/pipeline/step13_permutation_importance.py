"""
=========================================================
STEP 13

Permutation Importance
=========================================================
"""

import json
import joblib
import pandas as pd

from pathlib import Path

from src.explainability.permutation_importance import (
    PermutationImportance,
)


FEATURE_SET = "HYBRID"


def load_best_subset():

    with open(

        f"outputs/subset_selection/{FEATURE_SET}/best_subset.json"

    ) as f:

        info = json.load(f)

    subset = info["selected_subset"]

    filename = subset.lower() + ".csv"

    features = pd.read_csv(

        f"outputs/feature_selection/{FEATURE_SET}/{filename}"

    )["Feature"].tolist()

    return features


def main():

    print("=" * 70)
    print("STEP 13 : PERMUTATION IMPORTANCE")
    print("=" * 70)

    features = load_best_subset()

    X_test = pd.read_csv(

        f"data/processed/encoded/test_{FEATURE_SET.lower()}.csv"

    )

    y_test = pd.read_csv(

        "data/processed/test_target.csv"

    ).squeeze()

    y_test = y_test.map({

        "No": 0,

        "Yes": 1,

    })

    X_test = X_test[features]

    model = joblib.load(

        "outputs/models/best_model.pkl"

    )

    engine = PermutationImportance(

        model,

        n_repeats=30,

    )

    importance = engine.compute(

        X_test,

        y_test,

    )

    engine.save(

        importance,

    )

    print()

    print("=" * 70)

    print("TOP 10 FEATURES")

    print("=" * 70)

    print(

        importance.head(10)

    )

    print()

    print("Saved to")

    print(

        Path("outputs/explainability")

    )


if __name__ == "__main__":

    main()