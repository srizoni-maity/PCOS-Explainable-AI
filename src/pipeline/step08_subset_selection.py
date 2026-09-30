from pathlib import Path
import json

import pandas as pd

from src.training.subset_evaluator import SubsetEvaluator
from src.training.subset_selector import SubsetSelector
from src.core.target_encoder import TargetEncoder


FEATURE_SET = "HYBRID"


def main():

    print("=" * 70)
    print("STEP 08 : FEATURE SUBSET SELECTION")
    print("=" * 70)

    X = pd.read_csv(

        f"data/processed/encoded/train_{FEATURE_SET.lower()}.csv"

    )

    y = pd.read_csv(

        "data/processed/train_target.csv"

    ).iloc[:, 0]

    y = TargetEncoder().transform(y)

    evaluator = SubsetEvaluator()

    summary, detailed = evaluator.evaluate(

        FEATURE_SET,

        X,

        y

    )

    selector = SubsetSelector()

    ranking, winner = selector.select(summary)

    out = Path(

        f"outputs/subset_selection/{FEATURE_SET}"

    )

    out.mkdir(

        parents=True,

        exist_ok=True

    )

    ranking.to_csv(

        out / "subset_results.csv",

        index=False

    )

    detailed.to_csv(

        out / "subset_detailed_results.csv",

        index=False

    )

    with open(

        out / "best_subset.json",

        "w"

    ) as f:

        json.dump(

            {

                "feature_set": FEATURE_SET,

                "selected_subset": winner["Subset"],

                "num_features": int(

                    winner["NumFeatures"]

                ),

                "selection_rule": [

                    "Highest Mean ROC-AUC",

                    "Highest Mean MCC",

                    "Highest Mean PR-AUC",

                    "Highest Mean F1",

                    "Lowest Mean Brier",

                    "Fewest Features"

                ],

                "metrics": {

                    "MeanROC": float(winner["MeanROC"]),

                    "MeanPR": float(winner["MeanPR"]),

                    "MeanMCC": float(winner["MeanMCC"]),

                    "MeanF1": float(winner["MeanF1"]),

                    "MeanBrier": float(winner["MeanBrier"])

                }

            },

            f,

            indent=4

        )

    print()

    print(ranking)

    print()

    print("Winner")

    print(winner)

    print()

    print("Saved to")

    print(out)


if __name__ == "__main__":

    main()