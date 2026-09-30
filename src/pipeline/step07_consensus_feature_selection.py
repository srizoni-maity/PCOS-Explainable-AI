"""
=========================================================
STEP 07

CONSENSUS FEATURE SELECTION

Runs:

1. Mutual Information
2. ANOVA
3. Random Forest
4. LASSO
5. XGBoost

↓

Borda Count Ranking

↓

Saves

ranking.csv
top5.csv
top8.csv
top10.csv
top12.csv
all.csv

For

RAW
COMPOSITE
HYBRID

=========================================================
"""

from pathlib import Path
import pandas as pd

from src.feature_selection.consensus import ConsensusFeatureSelector
from src.core.target_encoder import TargetEncoder


# ---------------------------------------------------------


def save_feature_lists(ranking, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ranking.to_csv(
        output_dir / "ranking.csv",
        index=False
    )

    subsets = {

        "top5":5,

        "top8":8,

        "top10":10,

        "top12":12,

        "all":len(ranking)

    }

    for name,k in subsets.items():

        ranking.head(k)[["Feature"]].to_csv(

            output_dir / f"{name}.csv",

            index=False

        )


# ---------------------------------------------------------


def process_feature_set(name):

    print(f"\n{name}")

    X = pd.read_csv(

        f"data/processed/encoded/train_{name.lower()}.csv"

    )

    y = pd.read_csv(

        "data/processed/train_target.csv"

    ).iloc[:,0]

    y = TargetEncoder().transform(y)

    selector = ConsensusFeatureSelector()

    ranking = selector.run(

        X,

        y

    )

    output_dir = Path(

        f"outputs/feature_selection/{name}"

    )

    save_feature_lists(

        ranking,

        output_dir

    )

    print(

        f"Saved to {output_dir}"

    )


# ---------------------------------------------------------


def main():

    print("="*70)

    print("STEP 07 : CONSENSUS FEATURE SELECTION")

    print("="*70)

    for feature_set in [

        "RAW",

        "COMPOSITE",

        "HYBRID"

    ]:

        process_feature_set(

            feature_set

        )

    print()

    print("="*70)

    print("Finished")

    print("="*70)


# ---------------------------------------------------------

if __name__ == "__main__":

    main()