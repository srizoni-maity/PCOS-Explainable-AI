import pandas as pd

from src.models.logistic_model import LogisticModel
from src.training.optuna_search import OptunaSearch

from src.core.target_encoder import TargetEncoder

X = pd.read_csv(
    "data/processed/encoded/train_hybrid.csv"
)

y = pd.read_csv(
    "data/processed/train_target.csv"
).iloc[:, 0]

y = TargetEncoder().transform(y)

search = OptunaSearch(
    n_trials=10
)

study = search.optimize(

    LogisticModel(),

    X,

    y

)

print()

print("Best Score")

print(study.best_value)

print()

print("Best Parameters")

print(study.best_params)