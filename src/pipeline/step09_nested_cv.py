"""
=========================================================
STEP 09

Nested Cross Validation Runs
Logistic Regression
Random Forest
XGBoost
CatBoost

using Best Feature Subset

=========================================================
"""

from pathlib import Path
import json
import pandas as pd

from src.training.nested_cv import NestedCV
from src.models.factory import get_models


FEATURE_SET = "HYBRID"


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


def main():

    print("=" * 70)
    print("STEP 09 : NESTED CROSS VALIDATION")
    print("=" * 70)

    subset_name, features = load_best_subset()

    print(f"\nFeature Set : {FEATURE_SET}")
    print(f"Subset      : {subset_name}")
    print(f"Features    : {len(features)}")

    X = pd.read_csv(

        f"data/processed/encoded/train_{FEATURE_SET.lower()}.csv"

    )

    y = pd.read_csv(

        "data/processed/train_target.csv"

    ).squeeze()

    LABEL_MAP = {

        "No": 0,

        "Yes": 1

    }

    y = y.map(LABEL_MAP).astype(int)

    X = X[features]

    output_folder = Path(

        "outputs/nested_cv"

    )

    output_folder.mkdir(

        parents=True,

        exist_ok=True

    )

    summary_rows = []

    for model in get_models():

        print()

        print(model.name)

        print("-" * 40)

        engine = NestedCV(

            n_outer_splits=5,

            n_inner_trials=30,

            scoring="ROC_AUC"

        )

        results = engine.evaluate(

            model,

            X,

            y

        )

        fold_df = results["fold_results"]

        summary = results["summary"]
        
        best_params = results["best_params"]
        with open(

            output_folder /
            f"{model.name}_best_params.json",
            "w"

        ) as f:
            json.dump(
                best_params,
                f,
                indent=4
            )


        fold_df.to_csv(

            output_folder /

            f"{model.name}_folds.csv",

            index=False

        )

        summary.to_csv(

            output_folder /

            f"{model.name}_summary.csv",

            index=False

        )

        row = {

            "Model": model.name

        }

        for _, r in summary.iterrows():

            row[r["Metric"]] = r["Mean"]

        summary_rows.append(row)

        print(summary)

    leaderboard = pd.DataFrame(

        summary_rows

    )

    leaderboard = leaderboard.sort_values(

        "ROC_AUC",

        ascending=False

    )

    leaderboard.to_csv(

        output_folder /

        "leaderboard.csv",

        index=False

    )

    print()

    print("=" * 70)

    print("FINAL LEADERBOARD")

    print("=" * 70)

    print(

        leaderboard[

            [

                "Model",

                "ROC_AUC",

                "PR_AUC",

                "F1",

                "MCC"

            ]

        ]

    )

    print()

    print("Saved to")

    print(output_folder)


if __name__ == "__main__":

    main()