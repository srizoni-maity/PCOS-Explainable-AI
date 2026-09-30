import pandas as pd

from src.training.benchmark import Benchmark
from src.models.factory import get_models
from src.core.target_encoder import TargetEncoder

X = pd.read_csv(
    "data/processed/encoded/train_hybrid.csv"
)

y = pd.read_csv(
    "data/processed/train_target.csv"
).iloc[:,0]

y = TargetEncoder().transform(y)

engine = Benchmark()

results = engine.evaluate(

    get_models(),

    X,

    y

)

print(results)