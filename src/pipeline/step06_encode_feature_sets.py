from pathlib import Path

import pandas as pd

from src.core.encoder import FeatureEncoder


ROOT = Path("data/processed/feature_sets")

OUTPUT = Path("data/processed/encoded")

ENCODER_DIR = Path("outputs/models/encoders")


SETS = [

    "raw",

    "composite",

    "hybrid"

]


def process(name):

    train = pd.read_csv(

        ROOT / f"train_{name}.csv"

    )

    test = pd.read_csv(

        ROOT / f"test_{name}.csv"

    )

    encoder = FeatureEncoder()

    train_encoded = encoder.fit_transform(

        train

    )

    test_encoded = encoder.transform(

        test

    )

    OUTPUT.mkdir(

        parents=True,

        exist_ok=True

    )

    train_encoded.to_csv(

        OUTPUT /

        f"train_{name}.csv",

        index=False

    )

    test_encoded.to_csv(

        OUTPUT /

        f"test_{name}.csv",

        index=False

    )

    encoder.save(

        ENCODER_DIR /

        f"{name}_encoder.joblib"

    )

    print(

        f"{name.upper():10}"

        f" Train {train_encoded.shape}"

        f" Test {test_encoded.shape}"

    )


def main():

    print("=" * 70)

    print("LEAKAGE-FREE FEATURE ENCODING")

    print("=" * 70)

    for name in SETS:

        process(name)

    print("=" * 70)


if __name__ == "__main__":

    main()