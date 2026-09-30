"""
=========================================================
SHAP Report Generator

Creates

• Global Feature Importance CSV
• Report JSON
• Top Feature Summary

=========================================================
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


class SHAPReport:

    def __init__(self):

        pass

    # -----------------------------------------------------

    def create_feature_importance(

        self,

        shap_values,

        feature_names

    ):

        importance = np.abs(

            shap_values

        ).mean(axis=0)

        df = pd.DataFrame({

            "Feature": feature_names,

            "MeanAbsSHAP": importance

        })

        df = df.sort_values(

            "MeanAbsSHAP",

            ascending=False

        ).reset_index(drop=True)

        return df

    # -----------------------------------------------------

    def save_csv(

        self,

        importance_df,

        output_folder

    ):

        folder = Path(output_folder)

        folder.mkdir(

            parents=True,

            exist_ok=True

        )

        importance_df.to_csv(

            folder / "global_feature_importance.csv",

            index=False

        )

    # -----------------------------------------------------

    def save_json(

        self,

        importance_df,

        model_name,

        n_samples,

        output_folder

    ):

        folder = Path(output_folder)

        folder.mkdir(

            parents=True,

            exist_ok=True

        )

        report = {

            "Model": model_name,

            "Samples": int(n_samples),

            "NumFeatures": int(len(importance_df)),

            "TopFeature": importance_df.iloc[0]["Feature"],

            "TopFeatureMeanAbsSHAP": float(

                importance_df.iloc[0]["MeanAbsSHAP"]

            ),

            "Top10Features": (

                importance_df

                .head(10)

                .to_dict(

                    orient="records"

                )

            )

        }

        with open(

            folder / "report.json",

            "w"

        ) as f:

            json.dump(

                report,

                f,

                indent=4

            )

    # -----------------------------------------------------

    def generate(

        self,

        shap_values,

        feature_names,

        model_name,

        n_samples,

        output_folder

    ):

        importance_df = self.create_feature_importance(

            shap_values,

            feature_names

        )

        self.save_csv(

            importance_df,

            output_folder

        )

        self.save_json(

            importance_df,

            model_name,

            n_samples,

            output_folder

        )

        return importance_df