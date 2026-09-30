from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class DatasetProfiler:
    """
    Creates a complete profile of the cleaned dataset.

    This module NEVER modifies the data.
    """

    def __init__(self):

        self.profile = {}

    def profile_dataframe(self, df: pd.DataFrame):

        summary_rows = []

        for column in df.columns:

            unique_values = (
                df[column]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            unique_values = sorted(unique_values)

            self.profile[column] = {
                "dtype": str(df[column].dtype),
                "missing": int(df[column].isna().sum()),
                "unique_count": len(unique_values),
                "unique_values": unique_values
            }

            summary_rows.append({

                "Feature": column,

                "DataType": str(df[column].dtype),

                "MissingValues": int(df[column].isna().sum()),

                "UniqueValues": len(unique_values)

            })

        return pd.DataFrame(summary_rows)

    def save_profile(self,
                     report_path: Path,
                     summary_path: Path):

        report_path.parent.mkdir(parents=True, exist_ok=True)

        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:

            json.dump(
                self.profile,
                f,
                indent=4,
                ensure_ascii=False
            )

        summary = []

        for feature, info in self.profile.items():

            summary.append({

                "Feature": feature,

                "UniqueCount": info["unique_count"],

                "Missing": info["missing"]

            })

        pd.DataFrame(summary).to_csv(
            summary_path,
            index=False
        )