import pandas as pd

from src.feature_selection import (
    MutualInformationSelector,
    AnovaSelector,
    RandomForestSelector,
    LassoSelector,
    XGBoostSelector,
)

X = pd.read_csv("data/processed/encoded/train_hybrid.csv")

from src.core.target_encoder import TargetEncoder

y = pd.read_csv(
    "data/processed/train_target.csv"
).iloc[:, 0]

encoder = TargetEncoder()

y = encoder.transform(y)

selectors = [

    MutualInformationSelector(),
    AnovaSelector(),
    RandomForestSelector(),
    LassoSelector(),
    XGBoostSelector(),

]

for selector in selectors:

    scores = selector.fit(X, y)

    print(
        f"{selector.__class__.__name__:30}"
        f"{len(scores):3d} features"
        f"  Max={scores.max():.3f}"
    )