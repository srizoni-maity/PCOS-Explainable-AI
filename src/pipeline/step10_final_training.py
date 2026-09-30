from pathlib import Path
import json
import pandas as pd

from src.models.factory import get_models
from src.training.final_trainer import FinalTrainer


FEATURE_SET = "HYBRID"
WINNER = "CatBoost"


def load_best_subset():

    with open(

        f"outputs/subset_selection/{FEATURE_SET}/best_subset.json"

    ) as f:

        info = json.load(f)

    subset = info["selected_subset"]

    features = pd.read_csv(

        f"outputs/feature_selection/{FEATURE_SET}/{subset.lower()}.csv"

    )["Feature"].tolist()

    return features


def load_best_params():

    leaderboard = pd.read_csv(

        "outputs/nested_cv/leaderboard.csv"

    )

    winner = leaderboard.iloc[0]["Model"]

    with open(

        f"outputs/nested_cv/{winner}_best_params.json"

    ) as f:

        params = json.load(f)

    return winner, params


def main():

    print("=" * 70)
    print("STEP 10 : FINAL MODEL TRAINING")
    print("=" * 70)

    features = load_best_subset()

    X = pd.read_csv(

        f"data/processed/encoded/train_{FEATURE_SET.lower()}.csv"

    )[features]

    y = pd.read_csv(

        "data/processed/train_target.csv"

    ).squeeze()

    y = y.map({

        "No": 0,

        "Yes": 1

    }).astype(int)

    winner_name, params = load_best_params()

    model_definition = None

    for model in get_models():

        if model.name == winner_name:

            model_definition = model

            break

    trainer = FinalTrainer()

    trainer.train(

        model_definition,

        params,

        X,

        y

    )

    print()

    print("Winner")

    print(winner_name)

    print()

    print("Features")

    print(len(features))

    print()

    print("Training Samples")

    print(len(X))

    print()

    print("Saved")

    print("outputs/models")


if __name__ == "__main__":

    main()