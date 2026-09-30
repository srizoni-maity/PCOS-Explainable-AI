"""
=========================================================
Generic Benchmark Engine

Evaluates multiple models

using

5-fold Stratified CV

=========================================================
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .metrics import calculate_metrics


class Benchmark:

    def __init__(

        self,

        random_state=42

    ):

        self.cv = StratifiedKFold(

            n_splits=5,

            shuffle=True,

            random_state=random_state

        )

    # --------------------------------------------------

    def evaluate(

        self,

        models,

        X,

        y

    ):

        results = []

        for model_definition in models:

            model = model_definition.estimator()

            fold_metrics = []

            for train_idx,val_idx in self.cv.split(

                X,

                y

            ):

                estimator = clone(model)

                estimator.fit(

                    X.iloc[train_idx],

                    y.iloc[train_idx]

                )

                pred = estimator.predict(

                    X.iloc[val_idx]

                )

                prob = estimator.predict_proba(

                    X.iloc[val_idx]

                )[:,1]

                metrics = calculate_metrics(

                    y.iloc[val_idx],

                    pred,

                    prob

                )

                fold_metrics.append(

                    metrics

                )

            summary = {}

            for key in fold_metrics[0]:

                summary[key] = np.mean(

                    [

                        x[key]

                        for x in fold_metrics

                    ]

                )

            summary["Model"] = model_definition.name()

            results.append(

                summary

            )

        return pd.DataFrame(

            results

        )