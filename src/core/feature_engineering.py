"""
==========================================================
Feature Engineering Engine

Transforms the cleaned questionnaire into
machine-learning ready features.

Pipeline

Raw Questionnaire
        │
Risk Score Encoding
        │
Symptom Parsing
        │
Composite Features
        │
Validated Feature Dataset

==========================================================
"""

from __future__ import annotations

import pandas as pd

from src.core.risk_scores import RiskScorer
from src.core.symptom_parser import SymptomParser


class FeatureEngineer:

    """
    Main Feature Engineering Engine.
    """

    def __init__(self):

        self.scorer = RiskScorer()

    # -----------------------------------------------------
    # Ordinal Encoding
    # -----------------------------------------------------

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:

        data = df.copy()

        data["PackagedFoodScore"] = self.scorer.encode(
            data["PackagedFood"],
            self.scorer.PACKAGED_FOOD
        )

        data["FastFoodScore"] = self.scorer.encode(
            data["FastFood"],
            self.scorer.FAST_FOOD
        )

        data["FoodLabelScore"] = self.scorer.encode(
            data["FoodLabel"],
            self.scorer.FOOD_LABEL
        )

        data["ExerciseScore"] = self.scorer.encode(
            data["Exercise"],
            self.scorer.EXERCISE
        )

        data["SleepScore"] = self.scorer.encode(
            data["Sleep"],
            self.scorer.SLEEP
        )

        data["StressScore"] = self.scorer.encode(
            data["Stress"],
            self.scorer.STRESS
        )

        data["SugaryDrinkScore"] = self.scorer.encode(
            data["SugaryDrinks"],
            self.scorer.SUGARY_DRINK
        )

        data["PCOSAwarenessScore"] = self.scorer.encode(
            data["PCOSAwareness"],
            self.scorer.PCOS_AWARENESS
        )

        data["TransFatAwarenessScore"] = self.scorer.encode(
            data["TransFatAwareness"],
            self.scorer.TRANS_FAT_AWARENESS
        )

        data["MedicalRiskScore"] = data["MedicalHistory"].apply(
            self.scorer.medical_score
        )

        return data

    # -----------------------------------------------------
    # Symptom Features
    # -----------------------------------------------------

    def create_symptom_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        return SymptomParser.transform(df)

    # -----------------------------------------------------
    # Composite Features
    # -----------------------------------------------------

    def create_composite_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = df.copy()

        data["DietRiskScore"] = (

            data["PackagedFoodScore"] +

            data["FastFoodScore"] +

            data["SugaryDrinkScore"]

        )

        data["LifestyleRiskScore"] = (

            data["ExerciseScore"] +

            data["SleepScore"] +

            data["StressScore"]

        )

        data["AwarenessScore"] = (

            data["FoodLabelScore"] +

            data["PCOSAwarenessScore"] +

            data["TransFatAwarenessScore"]

        )

        return data

    # -----------------------------------------------------
    # Main Entry
    # -----------------------------------------------------

    def engineer(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = self.encode_features(df)

        data = self.create_symptom_features(data)

        data = self.create_composite_features(data)

        return data