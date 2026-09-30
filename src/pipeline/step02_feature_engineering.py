from pathlib import Path

import json

import pandas as pd

from src.core.feature_engineering import FeatureEngineer


INPUT = Path("data/processed/cleaned_dataset.csv")

OUTPUT = Path("data/processed/feature_engineered_dataset.csv")

REPORT = Path("outputs/reports/feature_engineering_report.json")


def main():

    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    engineer = FeatureEngineer()

    df = engineer.engineer(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT,
        index=False
    )

    report = {

        "rows": int(df.shape[0]),

        "columns": int(df.shape[1]),

        "engineered_features": [

            "DietRiskScore",
            "TransFatExposureScore",
            "LifestyleRiskScore",
            "SleepRiskScore",
            "StressRiskScore",
            "AwarenessScore",
            "MetabolicRiskScore",
            "HealthyLifestyleIndex",
            "OverallRiskScore"

        ]

    }

    with open(REPORT, "w") as f:
        json.dump(
            report,
            f,
            indent=4
        )

    print(f"Final Shape : {df.shape}")
    print(f"Saved Dataset : {OUTPUT}")
    print(f"Report : {REPORT}")

    print("=" * 70)


if __name__ == "__main__":
    main()