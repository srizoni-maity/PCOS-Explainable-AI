import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
)


def _positive_label(y):

    labels = list(np.unique(y))

    if 1 in labels:
        return 1

    if "Yes" in labels:
        return "Yes"

    if True in labels:
        return True

    return labels[-1]


def calculate_metrics(y_true, y_pred, y_prob):

    pos = _positive_label(y_true)

    return {

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            pos_label=pos,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            pos_label=pos,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            pos_label=pos,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            y_prob
        ),

        "PR_AUC": average_precision_score(
            y_true,
            y_prob
        ),

        "MCC": matthews_corrcoef(
            y_true,
            y_pred
        ),

        "Brier": brier_score_loss(
            y_true,
            y_prob
        )

    }