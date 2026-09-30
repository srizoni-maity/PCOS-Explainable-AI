import pandas as pd

from src.feature_selection.consensus import ConsensusFeatureSelector

from src.core.target_encoder import TargetEncoder

X = pd.read_csv(

    "data/processed/encoded/train_hybrid.csv"

)

y = pd.read_csv(

    "data/processed/train_target.csv"

).iloc[:,0]

y = TargetEncoder().transform(y)

selector = ConsensusFeatureSelector()

ranking = selector.run(

    X,

    y

)

print(ranking.head(15))