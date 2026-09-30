from pathlib import Path

import pandas as pd

from src.core.profiler import DatasetProfiler


INPUT = Path("data/processed/cleaned_dataset.csv")

PROFILE = Path("outputs/reports/feature_profile.json")

SUMMARY = Path("outputs/reports/feature_summary.csv")


def main():

    print("=" * 70)
    print("DATASET PROFILING")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    profiler = DatasetProfiler()

    summary_df = profiler.profile_dataframe(df)

    profiler.save_profile(PROFILE, SUMMARY)

    print(summary_df)

    print()

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print()

    print(f"Profile : {PROFILE}")
    print(f"Summary : {SUMMARY}")

    print("=" * 70)


if __name__ == "__main__":
    main()