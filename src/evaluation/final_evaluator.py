"""
=========================================================
Final Test Evaluation
=========================================================
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)


class FinalEvaluator:

    def __init__(self):

        self.output = Path("outputs/final_evaluation")

        self.figures = self.output / "figures"

        self.tables = self.output / "tables"

        self.figures.mkdir(parents=True, exist_ok=True)

        self.tables.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------

    def load_model(self):

        return joblib.load(
            "outputs/models/best_model.pkl"
        )

    # --------------------------------------------------

    def evaluate(self, X_test, y_test):

        model = self.load_model()

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred
        ).ravel()

        specificity = tn / (tn + fp)

        npv = tn / (tn + fn)

        metrics = {

            "Accuracy": accuracy_score(y_test, y_pred),

            "Precision": precision_score(y_test, y_pred),

            "Recall": recall_score(y_test, y_pred),

            "Specificity": specificity,

            "F1": f1_score(y_test, y_pred),

            "ROC_AUC": roc_auc_score(y_test, y_prob),

            "PR_AUC": average_precision_score(y_test, y_prob),

            "MCC": matthews_corrcoef(y_test, y_pred),

            "Brier": brier_score_loss(y_test, y_prob),

            "BalancedAccuracy": balanced_accuracy_score(
                y_test,
                y_pred
            ),

            "CohenKappa": cohen_kappa_score(
                y_test,
                y_pred
            ),

            "PPV": precision_score(y_test, y_pred),

            "NPV": npv,

            "FPR": fp / (fp + tn),

            "FNR": fn / (fn + tp),

        }

        metrics_df = pd.DataFrame(
            metrics.items(),
            columns=["Metric", "Value"]
        )

        metrics_df.to_csv(
            self.tables / "metrics.csv",
            index=False
        )

        cm_df = pd.DataFrame(

            confusion_matrix(y_test, y_pred),

            index=["Actual_No", "Actual_Yes"],

            columns=["Pred_No", "Pred_Yes"]

        )

        cm_df.to_csv(
            self.tables / "confusion_matrix.csv"
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        pd.DataFrame(report).transpose().to_csv(
            self.tables / "classification_report.csv"
        )

        predictions = pd.DataFrame({

            "TrueLabel": y_test,

            "PredictedLabel": y_pred,

            "Probability": y_prob

        })

        predictions.to_csv(
            self.tables / "predictions.csv",
            index=False
        )

        return {

            "metrics": metrics_df,

            "y_true": y_test,

            "y_pred": y_pred,

            "y_prob": y_prob,

            "cm": cm_df

        }