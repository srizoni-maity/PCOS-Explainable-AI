"""
=========================================================
SHAP Plot Generator

Generates publication-quality SHAP visualizations.

Outputs
--------
• Summary Beeswarm
• Summary Bar
• Dependence Plots
• Waterfall Plots
• Force Plot (HTML)

=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class SHAPPlots:

    def __init__(self, dpi=300):

        self.dpi = dpi

    # -----------------------------------------------------

    def _prepare_folder(self, folder):

        folder = Path(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        return folder

    # -----------------------------------------------------

    def summary_beeswarm(

        self,

        shap_values,

        X,

        output_folder

    ):

        folder = self._prepare_folder(output_folder)

        plt.figure(figsize=(10, 7))

        shap.summary_plot(

            shap_values,

            X,

            show=False

        )

        plt.tight_layout()

        plt.savefig(

            folder / "summary_beeswarm.png",

            dpi=self.dpi,

            bbox_inches="tight"

        )

        plt.close()

    # -----------------------------------------------------

    def summary_bar(

        self,

        shap_values,

        X,

        output_folder

    ):

        folder = self._prepare_folder(output_folder)

        plt.figure(figsize=(8, 7))

        shap.summary_plot(

            shap_values,

            X,

            plot_type="bar",

            show=False

        )

        plt.tight_layout()

        plt.savefig(

            folder / "summary_bar.png",

            dpi=self.dpi,

            bbox_inches="tight"

        )

        plt.close()

    # -----------------------------------------------------

    def dependence_plots(

        self,

        shap_values,

        X,

        output_folder,

        top_k=10

    ):

        folder = self._prepare_folder(

            Path(output_folder) / "dependence"

        )

        importance = np.abs(shap_values).mean(axis=0)

        ranking = pd.Series(

            importance,

            index=X.columns

        ).sort_values(

            ascending=False

        )

        for feature in ranking.head(top_k).index:

            plt.figure(figsize=(7, 5))

            shap.dependence_plot(

                feature,

                shap_values,

                X,

                show=False

            )

            plt.tight_layout()

            plt.savefig(

                folder / f"{feature}.png",

                dpi=self.dpi,

                bbox_inches="tight"

            )

            plt.close()

    # -----------------------------------------------------

    def waterfall_plots(

        self,

        explainer,

        shap_values,

        X,

        output_folder,

        n_patients=5

    ):

        folder = self._prepare_folder(

            Path(output_folder) / "waterfall"

        )

        n_patients = min(

            n_patients,

            len(X)

        )

        for i in range(n_patients):

            explanation = shap.Explanation(

                values=shap_values[i],

                base_values=explainer.expected_value,

                data=X.iloc[i],

                feature_names=X.columns

            )

            plt.figure(figsize=(9, 6))

            shap.plots.waterfall(

                explanation,

                show=False

            )

            plt.savefig(

                folder / f"patient_{i+1:03d}.png",

                dpi=self.dpi,

                bbox_inches="tight"

            )

            plt.close()

    # -----------------------------------------------------

    def force_plot(

        self,

        explainer,

        shap_values,

        X,

        output_folder,

        patient=0

    ):

        folder = self._prepare_folder(

            Path(output_folder) / "force"

        )

        force = shap.force_plot(

            explainer.expected_value,

            shap_values[patient],

            X.iloc[patient],

            matplotlib=False

        )

        shap.save_html(

            str(

                folder /

                f"patient_{patient+1:03d}.html"

            ),

            force

        )