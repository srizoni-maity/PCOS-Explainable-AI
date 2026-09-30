"""
Target Encoder

Encodes the target variable once for the
entire ML pipeline.
"""

import pandas as pd


class TargetEncoder:

    mapping = {

        "No": 0,
        "Yes": 1

    }

    inverse_mapping = {

        0: "No",
        1: "Yes"

    }

    def transform(self, y):

        if isinstance(y, pd.Series):

            return y.map(self.mapping).astype(int)

        return pd.Series(y).map(self.mapping).astype(int)

    def inverse_transform(self, y):

        if isinstance(y, pd.Series):

            return y.map(self.inverse_mapping)

        return pd.Series(y).map(self.inverse_mapping)