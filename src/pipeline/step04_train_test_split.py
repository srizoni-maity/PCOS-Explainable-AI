from pathlib import Path

import pandas as pd

from src.core.feature_engineering import FeatureEngineer
from src.core.splitter import DatasetSplitter


INPUT = Path("data/processed/cleaned_dataset.csv")


def main():

    print("=" * 70)
    print("LEAKAGE-FREE TRAIN / TEST SPLIT")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    engineer = FeatureEngineer()

    df = engineer.engineer(df)

    splitter = DatasetSplitter()

    X_train, X_test, y_train, y_test = splitter.split(df)

    splitter.save(

        X_train,
        X_test,

        y_train,
        y_test

    )

    splitter.report(

        X_train,
        X_test,

        y_train,
        y_test

    )

    print(f"Training Samples : {len(X_train)}")

    print(f"Testing Samples  : {len(X_test)}")

    print()

    print(f"Training Features : {X_train.shape}")

    print(f"Testing Features  : {X_test.shape}")

    print()

    print("Saved")

    print("  train_features.csv")

    print("  test_features.csv")

    print("  train_target.csv")

    print("  test_target.csv")

    print()

    print("Report")

    print("  outputs/reports/split_report.json")

    print("=" * 70)


if __name__ == "__main__":
    main()