from pathlib import Path

import pandas as pd

from src.core.feature_sets import FeatureSetBuilder

TRAIN = Path("data/processed/train_features.csv")
TEST = Path("data/processed/test_features.csv")

OUTPUT = Path("data/processed/feature_sets")


def main():

    print("=" * 70)
    print("CREATING FEATURE SETS")
    print("=" * 70)

    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)

    builder = FeatureSetBuilder()

    datasets = {

        "raw": (

            builder.build_raw(train),

            builder.build_raw(test)

        ),

        "composite": (

            builder.build_composite(train),

            builder.build_composite(test)

        ),

        "hybrid": (

            builder.build_hybrid(train),

            builder.build_hybrid(test)

        )

    }

    for name, (tr, te) in datasets.items():

        builder.save(

            tr,
            te,

            OUTPUT,

            name

        )

        print(f"{name.upper():10} Train {tr.shape}  Test {te.shape}")

    print("=" * 70)


if __name__ == "__main__":
    main()