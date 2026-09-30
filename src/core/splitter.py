"""
==========================================================
Leakage-Free Train/Test Splitter
==========================================================
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplitter:

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42
    ):

        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame):

        X = df.drop(columns=["PCOS"])

        y = df["PCOS"]

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=self.test_size,

            stratify=y,

            shuffle=True,

            random_state=self.random_state

        )

        return X_train, X_test, y_train, y_test

    @staticmethod
    def save(

        X_train,
        X_test,
        y_train,
        y_test,

        output_dir="data/processed"

    ):

        output = Path(output_dir)

        output.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(output / "train_features.csv", index=False)

        X_test.to_csv(output / "test_features.csv", index=False)

        y_train.to_csv(output / "train_target.csv", index=False)

        y_test.to_csv(output / "test_target.csv", index=False)

    @staticmethod
    def report(

        X_train,
        X_test,
        y_train,
        y_test,

        report_path="outputs/reports/split_report.json"

    ):

        Path(report_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        report = {

            "training_samples": int(len(X_train)),

            "testing_samples": int(len(X_test)),

            "features": int(X_train.shape[1]),

            "training_positive": int((y_train == "Yes").sum()),

            "training_negative": int((y_train == "No").sum()),

            "testing_positive": int((y_test == "Yes").sum()),

            "testing_negative": int((y_test == "No").sum()),

            "test_size": 0.20,

            "random_state": 42

        }

        with open(report_path, "w") as f:

            json.dump(report, f, indent=4)