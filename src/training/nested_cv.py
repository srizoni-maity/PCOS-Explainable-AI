"""
=========================================================
Nested Cross Validation Engine

Outer CV
    ↓
Inner Optuna Search
    ↓
Best Hyperparameters
    ↓
Train Final Model
    ↓
Evaluate Outer Fold

Returns
-------
fold_results
summary

=========================================================
"""

import time
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .trainer import Trainer
from .optuna_search import OptunaSearch
from .metrics import calculate_metrics
from .model_result import FoldResult


class NestedCV:

    def __init__(

        self,

        n_outer_splits=5,

        n_inner_trials=30,

        scoring="ROC_AUC",

        random_state=42

    ):

        self.outer_cv = StratifiedKFold(

            n_splits=n_outer_splits,

            shuffle=True,

            random_state=random_state

        )

        self.search = OptunaSearch(

            n_trials=n_inner_trials,

            random_state=random_state

        )

        self.trainer = Trainer()

        self.scoring = scoring

    # ---------------------------------------------------------

    def evaluate(

        self,

        model_definition,

        X,

        y

    ):

        fold_results = []

        for fold, (train_idx, valid_idx) in enumerate(

            self.outer_cv.split(X, y),

            start=1

        ):

            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]

            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            # ---------------------------------------------
            # Hyperparameter Optimization
            # ---------------------------------------------

            search_result = self.search.optimize(

                model_definition=model_definition,

                X=X_train,

                y=y_train,

                scoring=self.scoring

            )

            best_params = search_result["best_params"]

            # ---------------------------------------------
            # Train
            # ---------------------------------------------

            start_train = time.perf_counter()

            estimator = self.trainer.train(

                model_definition.estimator(),

                best_params,

                X_train,

                y_train

            )

            train_time = (

                time.perf_counter()

                - start_train

            )

            # ---------------------------------------------
            # Prediction
            # ---------------------------------------------

            start_predict = time.perf_counter()

            y_pred = estimator.predict(

                X_valid

            )

            y_prob = estimator.predict_proba(

                X_valid

            )[:, 1]

            inference_time = (

                time.perf_counter()

                - start_predict

            )

            metrics = calculate_metrics(

                y_valid,

                y_pred,

                y_prob

            )

            fold_results.append(

                FoldResult(

                    model_name=model_definition.name,

                    fold=fold,

                    accuracy=metrics["Accuracy"],

                    precision=metrics["Precision"],

                    recall=metrics["Recall"],

                    f1=metrics["F1"],

                    roc_auc=metrics["ROC_AUC"],

                    pr_auc=metrics["PR_AUC"],

                    mcc=metrics["MCC"],

                    brier=metrics["Brier"],

                    train_time=train_time,

                    inference_time=inference_time,

                    best_params=best_params

                ).__dict__

            )

        fold_df = pd.DataFrame(

            fold_results

        )

        summary = pd.DataFrame({

            "Metric": [

                "Accuracy",

                "Precision",

                "Recall",

                "F1",

                "ROC_AUC",

                "PR_AUC",

                "MCC",

                "Brier",

                "TrainTime",

                "InferenceTime"

            ],

            "Mean": [

                fold_df["accuracy"].mean(),

                fold_df["precision"].mean(),

                fold_df["recall"].mean(),

                fold_df["f1"].mean(),

                fold_df["roc_auc"].mean(),

                fold_df["pr_auc"].mean(),

                fold_df["mcc"].mean(),

                fold_df["brier"].mean(),

                fold_df["train_time"].mean(),

                fold_df["inference_time"].mean()

            ],

            "Std": [

                fold_df["accuracy"].std(),

                fold_df["precision"].std(),

                fold_df["recall"].std(),

                fold_df["f1"].std(),

                fold_df["roc_auc"].std(),

                fold_df["pr_auc"].std(),

                fold_df["mcc"].std(),

                fold_df["brier"].std(),

                fold_df["train_time"].std(),

                fold_df["inference_time"].std()

            ]

        })
        
        # ---------------------------------------------------------
        # Select Best Fold (highest ROC)
        # ---------------------------------------------------------

        best_fold = fold_df.sort_values(
            "roc_auc",
            ascending=False
        ).iloc[0]

        return {

            "fold_results": fold_df,

            "summary": summary,

            "best_params": best_fold["best_params"]

        }
        