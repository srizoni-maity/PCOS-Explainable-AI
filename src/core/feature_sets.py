"""
=========================================================
Feature Set Builder

Creates three feature representations:

1. RAW
2. COMPOSITE
3. HYBRID

=========================================================
"""

from pathlib import Path
import pandas as pd


class FeatureSetBuilder:

    # -----------------------------
    # Binary symptom features
    # -----------------------------

    SYMPTOM_COLUMNS = [

        "Symptom_IrregularCycle",
        "Symptom_WeightGain",
        "Symptom_Acne",
        "Symptom_Hirsutism",
        "Symptom_None",
        "SymptomCount"

    ]

    # -----------------------------
    # Numeric encoded features
    # -----------------------------

    SCORE_COLUMNS = [

        "PackagedFoodScore",
        "FastFoodScore",
        "FoodLabelScore",
        "ExerciseScore",
        "SleepScore",
        "StressScore",
        "SugaryDrinkScore",
        "PCOSAwarenessScore",
        "TransFatAwarenessScore",
        "MedicalRiskScore"

    ]

    # -----------------------------
    # Composite Features
    # -----------------------------

    COMPOSITE_COLUMNS = [

        "DietRiskScore",
        "LifestyleRiskScore",
        "AwarenessScore"

    ]

    @staticmethod
    def build_raw(df: pd.DataFrame):

        cols = [

            "AgeGroup",
            "Student",
            "Residence",
            "PackagedFood",
            "FastFood",
            "FoodLabel",
            "Exercise",
            "Sleep",
            "Stress",
            "SugaryDrinks",
            "PCOSAwareness",
            "TransFatAwareness",
            "MedicalHistory"

        ]

        cols += FeatureSetBuilder.SYMPTOM_COLUMNS

        return df[cols].copy()

    @staticmethod
    def build_composite(df):

        cols = list(
            FeatureSetBuilder.build_raw(df).columns
        )

        cols += FeatureSetBuilder.SCORE_COLUMNS

        return df[cols].copy()

    @staticmethod
    def build_hybrid(df):

        cols = list(
            FeatureSetBuilder.build_composite(df).columns
        )

        cols += FeatureSetBuilder.COMPOSITE_COLUMNS

        return df[cols].copy()

    @staticmethod
    def save(train_df,
             test_df,
             folder,
             name):

        folder = Path(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        train_df.to_csv(

            folder / f"train_{name}.csv",

            index=False

        )

        test_df.to_csv(

            folder / f"test_{name}.csv",

            index=False

        )