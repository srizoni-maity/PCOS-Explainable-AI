"""
==========================================================
Leakage-Free Feature Encoder

Fits ONLY on training data.

Transforms both train and test
using the same encoder.

==========================================================
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


class FeatureEncoder:

    def __init__(self):

        self.encoder = None

        self.categorical_columns = None

        self.numeric_columns = None

    # --------------------------------------------------

    def detect_columns(self, X):

        self.categorical_columns = list(

            X.select_dtypes(

                include=["object"]

            ).columns

        )

        self.numeric_columns = list(

            X.select_dtypes(

                exclude=["object"]

            ).columns

        )

    # --------------------------------------------------

    def fit(self, X):

        self.detect_columns(X)

        self.encoder = ColumnTransformer(

            transformers=[

                (

                    "categorical",

                    OrdinalEncoder(

                        handle_unknown="use_encoded_value",

                        unknown_value=-1

                    ),

                    self.categorical_columns

                )

            ],

            remainder="passthrough"

        )

        self.encoder.fit(X)

        return self

    # --------------------------------------------------

    def transform(self, X):

        data = self.encoder.transform(X)

        columns = (

            self.categorical_columns +

            self.numeric_columns

        )

        return pd.DataFrame(

            data,

            columns=columns,

            index=X.index

        )

    # --------------------------------------------------

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)

    # --------------------------------------------------

    def save(self, path):

        Path(path).parent.mkdir(

            parents=True,

            exist_ok=True

        )

        joblib.dump(

            self.encoder,

            path

        )

    # --------------------------------------------------

    def load(self, path):

        self.encoder = joblib.load(path)
