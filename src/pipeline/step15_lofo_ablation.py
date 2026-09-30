"""
=========================================================
STEP 15

Leave-One-Feature-Group-Out Ablation Study

IEEE Version

=========================================================
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.lofo_ablation import LOFOAblation
from src.models.catboost_model import CatBoostModel


FEATURE_SET = "HYBRID"


# ----------------------------------------------------------


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


# ----------------------------------------------------------


def plot_roc_drop(drop_df, output_folder):

    df = drop_df.copy()

    df = df[df["Experiment"] != "Full"]

    plt.figure(figsize=(9,5))

    plt.bar(

        df["Experiment"],

        -df["DeltaROC_AUC"]

    )

    plt.ylabel("ROC-AUC Drop")

    plt.title("Performance Drop After Removing Feature Groups")

    plt.tight_layout()

    plt.savefig(

        output_folder /

        "roc_drop.png",

        dpi=600

    )

    plt.close()


# ----------------------------------------------------------


def plot_heatmap(comparison, output_folder):

    metrics = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "ROC_AUC",

        "PR_AUC",

        "MCC"

    ]

    matrix = comparison.set_index(

        "Experiment"

    )[metrics]

    plt.figure(figsize=(9,5))

    plt.imshow(

        matrix,

        aspect="auto"

    )

    plt.xticks(

        range(len(metrics)),

        metrics,

        rotation=45,

        ha="right"

    )

    plt.yticks(

        range(len(matrix.index)),

        matrix.index

    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(

        output_folder /

        "heatmap.png",

        dpi=600

    )

    plt.close()


# ----------------------------------------------------------


def plot_radar(comparison, output_folder):

    import numpy as np

    metrics = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "ROC_AUC",

        "PR_AUC",

        "MCC"

    ]

    angles = np.linspace(

        0,

        2*np.pi,

        len(metrics),

        endpoint=False

    ).tolist()

    angles += angles[:1]

    fig = plt.figure(figsize=(8,8))

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

            label=row["Experiment"]

        )

        ax.fill(

            angles,

            values,

            alpha=0.08

        )

    ax.set_xticks(

        angles[:-1]

    )

    ax.set_xticklabels(

        metrics

    )

    ax.set_ylim(

        0,

        1

    )

    ax.legend(

        loc="upper right"

    )

    plt.tight_layout()

    plt.savefig(

        output_folder /

        "radar.png",

        dpi=600

    )

    plt.close()


# ----------------------------------------------------------


def save_outputs(results, drop_df, folder):

    results["comparison"].to_csv(

        folder /

        "comparison.csv",

        index=False

    )

    drop_df.to_csv(

        folder /

        "roc_drop.csv",

        index=False

    )

    for name, df in results["details"].items():

        df.to_csv(

            folder /

            f"{name}.csv",

            index=False

        )

    with open(

        folder /

        "feature_groups.json",

        "w"

    ) as f:

        json.dump(

            results["groups"],

            f,

            indent=4

        )


# ----------------------------------------------------------


def main():

    print("="*70)

    print("STEP 15 : LOFO ABLATION STUDY")

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

    LABEL_MAP = {

        "No":0,

        "Yes":1

    }

    y_train = y_train.map(

        LABEL_MAP

    ).astype(int)

    y_test = y_test.map(

        LABEL_MAP

    ).astype(int)

    X_train = X_train[features]

    X_test = X_test[features]

    model = CatBoostModel()

    engine = LOFOAblation()

    results = engine.evaluate(

        model,

        X_train,

        y_train,

        X_test,

        y_test

    )

    drop = engine.performance_drop(

        results["comparison"]

    )

    output_folder = Path(

        "outputs/lofo_ablation"

    )

    output_folder.mkdir(

        parents=True,

        exist_ok=True

    )

    save_outputs(

        results,

        drop,

        output_folder

    )

    plot_roc_drop(

        drop,

        output_folder

    )

    plot_heatmap(

        results["comparison"],

        output_folder

    )

    plot_radar(

        results["comparison"],

        output_folder

    )

    print()

    print("="*70)

    print("LOFO RESULTS")

    print("="*70)

    print(

        results["comparison"][

            [

                "Experiment",

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

    print("="*70)

    print("PERFORMANCE DROP")

    print("="*70)

    print(drop)

    print()

    print("Saved to")

    print(output_folder)


# ----------------------------------------------------------

if __name__ == "__main__":

    main()