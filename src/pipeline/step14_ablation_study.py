"""
=========================================================
STEP 14

Ablation Study

Clinical vs Lifestyle vs Full

=========================================================
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.ablation import AblationStudy
from src.models.catboost_model import CatBoostModel

FEATURE_SET = "HYBRID"


# ---------------------------------------------------------


def load_best_subset():

    with open(

        f"outputs/subset_selection/{FEATURE_SET}/best_subset.json"

    ) as f:

        info = json.load(f)

    subset = info["selected_subset"]

    filename = subset.lower() + ".csv"

    features = pd.read_csv(

        f"outputs/feature_selection/{FEATURE_SET}/{filename}"

    )["Feature"].tolist()

    return subset, features


# ---------------------------------------------------------


def save_results(

    results,

    output_folder

):

    comparison = results["comparison"]

    comparison.to_csv(

        output_folder /

        "comparison.csv",

        index=False

    )

    for name, df in results["details"].items():

        df.to_csv(

            output_folder /

            f"{name.lower()}_metrics.csv",

            index=False

        )

    groups = results["groups"]

    with open(

        output_folder /

        "feature_groups.json",

        "w"

    ) as f:

        json.dump(

            groups,

            f,

            indent=4

        )


# ---------------------------------------------------------


def plot_bar(

    comparison,

    output_folder

):

    plt.figure(

        figsize=(8,6)

    )

    plt.bar(

        comparison["Model"],

        comparison["ROC_AUC"]

    )

    plt.ylabel(

        "ROC-AUC"

    )

    plt.title(

        "Ablation Study"

    )

    plt.tight_layout()

    plt.savefig(

        output_folder /

        "ablation_bar.png",

        dpi=600

    )

    plt.close()


# ---------------------------------------------------------


def plot_radar(

    comparison,

    output_folder

):

    metrics = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "ROC_AUC",

        "PR_AUC",

        "MCC"

    ]

    import numpy as np

    angles = np.linspace(

        0,

        2*np.pi,

        len(metrics),

        endpoint=False

    ).tolist()

    angles += angles[:1]

    fig = plt.figure(

        figsize=(8,8)

    )

    ax = plt.subplot(

        polar=True

    )

    for _, row in comparison.iterrows():

        values = [

            row[m]

            for m in metrics

        ]

        values += values[:1]

        ax.plot(

            angles,

            values,

            linewidth=2,

            label=row["Model"]

        )

        ax.fill(

            angles,

            values,

            alpha=0.10

        )

    ax.set_xticks(

        angles[:-1]

    )

    ax.set_xticklabels(

        metrics

    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(

        output_folder /

        "ablation_radar.png",

        dpi=600

    )

    plt.close()


# ---------------------------------------------------------


def main():

    print("="*70)

    print("STEP 14 : ABLATION STUDY")

    print("="*70)

    subset, features = load_best_subset()

    print()

    print("Feature Set :", FEATURE_SET)

    print("Subset      :", subset)

    print("Features    :", len(features))

    X_train = pd.read_csv(

        f"data/processed/encoded/train_{FEATURE_SET.lower()}.csv"

    )

    X_test = pd.read_csv(

        f"data/processed/encoded/test_{FEATURE_SET.lower()}.csv"

    )

    y_train = pd.read_csv(

        "data/processed/train_target.csv"

    ).squeeze()

    y_test = pd.read_csv(

        "data/processed/test_target.csv"

    ).squeeze()

    label_map = {

        "No":0,

        "Yes":1

    }

    y_train = y_train.map(

        label_map

    ).astype(int)

    y_test = y_test.map(

        label_map

    ).astype(int)

    X_train = X_train[features]

    X_test = X_test[features]

    engine = AblationStudy()

    model = CatBoostModel()

    results = engine.evaluate(

        model,

        X_train,

        y_train,

        X_test,

        y_test

    )

    output_folder = Path(

        "outputs/ablation"

    )

    output_folder.mkdir(

        parents=True,

        exist_ok=True

    )

    save_results(

        results,

        output_folder

    )

    plot_bar(

        results["comparison"],

        output_folder

    )

    plot_radar(

        results["comparison"],

        output_folder

    )

    print()

    print("="*70)

    print("ABLATION RESULTS")

    print("="*70)

    print(

        results["comparison"][

            [

                "Model",

                "NumFeatures",

                "Accuracy",

                "ROC_AUC",

                "PR_AUC",

                "F1",

                "MCC"

            ]

        ]

    )

    print()

    print("Saved to")

    print(output_folder)


# ---------------------------------------------------------

if __name__ == "__main__":

    main()