from dataclasses import dataclass


@dataclass
class FoldResult:

    model_name: str

    fold: int

    accuracy: float

    precision: float

    recall: float

    f1: float

    roc_auc: float

    pr_auc: float

    mcc: float

    brier: float

    train_time: float

    inference_time: float

    best_params: dict