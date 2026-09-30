"""
=========================================================
Permutation Importance
=========================================================
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


class PermutationImportance:

    def __init__(
        self,
        model,
        random_state=42,
        n_repeats=30,
    ):

        self.model = model
        self.random_state = random_state
        self.n_repeats = n_repeats

    # -----------------------------------------------------

    def compute(
        self,
        X,
        y,
    ):

        result = permutation_importance(

            self.model,

            X,

            y,

            scoring="roc_auc",

            n_repeats=self.n_repeats,

            random_state=self.random_state,

            n_jobs=-1,

        )

        importance = pd.DataFrame({

            "Feature": X.columns,

            "Importance": result.importances_mean,

            "Std": result.importances_std,

        })

        importance = importance.sort_values(

            "Importance",

            ascending=False,

        ).reset_index(drop=True)

        return importance

    # -----------------------------------------------------

    def save(
        self,
        importance,
        folder="outputs/explainability",
    ):

        folder = Path(folder)

        folder.mkdir(

            parents=True,

            exist_ok=True,

        )

        importance.to_csv(

            folder / "permutation_importance.csv",

            index=False,

        )

        plt.figure(figsize=(9,7))

        plt.barh(

            importance["Feature"][:20][::-1],

            importance["Importance"][:20][::-1]

        )

        plt.xlabel("Permutation Importance")

        plt.tight_layout()

        plt.savefig(

            folder / "permutation_importance.png",

            dpi=300,

            bbox_inches="tight",

        )

        plt.close()