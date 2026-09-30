"""
==========================================================
Risk Score Engine
----------------------------------------------------------
Centralized scoring rules for all questionnaire variables.

IEEE Research Project
PCOS Explainable Machine Learning Framework

This module MUST NOT depend on dataset statistics.

Only deterministic row-wise transformations.

==========================================================
"""

from __future__ import annotations


class RiskScorer:

    # ---------------------------------------------------
    # Dietary Behaviour
    # ---------------------------------------------------

    PACKAGED_FOOD = {
        "Rarely": 0,
        "1-2 times per week": 1,
        "3-5 times per week": 2,
        "Daily": 3
    }

    FAST_FOOD = {
        "Rarely/Never": 0,
        "Occasionally": 1,
        "Several times per week": 2,
        "Daily": 3
    }

    SUGARY_DRINK = {
        "Rarely/Never": 0,
        "Occasionally": 1,
        "Several times per week": 2,
        "Daily": 3
    }

    # ---------------------------------------------------
    # Food Label Reading
    # ---------------------------------------------------

    FOOD_LABEL = {
        "Always": 0,
        "Often": 1,
        "Sometimes": 2,
        "Never": 3
    }

    # ---------------------------------------------------
    # Awareness
    # ---------------------------------------------------

    TRANS_FAT_AWARENESS = {

        "Strongly agree": 0,
        "Agree": 1,
        "Disagree": 2,
        "Strongly disagree": 3

    }

    PCOS_AWARENESS = {

        "Very familiar": 0,
        "Somehow familiar": 1,
        "Not familiar at all": 2

    }

    # ---------------------------------------------------
    # Exercise
    # ---------------------------------------------------

    EXERCISE = {

        "Daily": 0,
        "3-5 times per week": 1,
        "1-2 times per week": 2,
        "Rarely/Never": 3

    }

    # ---------------------------------------------------
    # Sleep
    # ---------------------------------------------------

    SLEEP = {

        "7-8 hours": 0,
        "More than 8 hours": 1,
        "5-6 hours": 2,
        "Less than 5 hours": 3

    }

    # ---------------------------------------------------
    # Stress
    # ---------------------------------------------------

    STRESS = {

        "Very low": 0,
        "Low": 1,
        "Moderate": 2,
        "Very high": 3

    }

    # ---------------------------------------------------
    # Medical History
    # ---------------------------------------------------

    @staticmethod
    def medical_score(text: str) -> int:

        if not isinstance(text, str):
            return 0

        score = 0

        if "Obesity" in text:
            score += 2

        if "High blood sugar" in text:
            score += 2

        if "High cholesterol" in text:
            score += 1

        return score

    # ---------------------------------------------------
    # Symptom Count
    # ---------------------------------------------------

    @staticmethod
    def symptom_count(text: str) -> int:

        if not isinstance(text, str):
            return 0

        if text == "None above":
            return 0

        return len(text.split(";"))

    # ---------------------------------------------------
    # Generic Mapper
    # ---------------------------------------------------

    @staticmethod
    def encode(series, mapping):

        unknown = set(series.dropna().unique()) - set(mapping.keys())

        if unknown:
            raise ValueError(
                f"Unknown categories found:\n{unknown}"
            )

        return series.map(mapping)

    # ---------------------------------------------------
    # Validation
    # ---------------------------------------------------

    @staticmethod
    def validate_scores(df):

        if df.isna().sum().sum() != 0:
            raise ValueError(
                "Feature engineering produced NaN values."
            )

        return True