import pandas as pd

from src.core.feature_engineering import FeatureEngineer

df = pd.read_csv(
    "data/processed/cleaned_dataset.csv"
)

engineer = FeatureEngineer()

df = engineer.engineer(df)

print(df.shape)

print()

print(df.columns.tolist())

print()

print(df.head())