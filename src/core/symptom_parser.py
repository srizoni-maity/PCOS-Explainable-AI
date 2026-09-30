"""
=========================================================
Symptom Parser

Converts multi-select symptom strings into binary features.

Example:

Weight gain;Acne or oily skin

↓

WeightGain = 1
Acne = 1
IrregularCycle = 0
Hirsutism = 0
SymptomCount = 2

=========================================================
"""

from __future__ import annotations

import pandas as pd


class SymptomParser:

    SYMPTOMS = {

        "Irregular menstrual cycle":
            "Symptom_IrregularCycle",

        "Weight gain":
            "Symptom_WeightGain",

        "Acne or oily skin":
            "Symptom_Acne",

        "Excess facial/body hair":
            "Symptom_Hirsutism"

    }

    @classmethod
    def transform(cls, df: pd.DataFrame):

        data = df.copy()

        # initialize binary columns

        for feature in cls.SYMPTOMS.values():

            data[feature] = 0

        data["Symptom_None"] = 0

        data["SymptomCount"] = 0

        # ---------------------------------------------

        for idx, value in data["Symptoms"].items():

            if pd.isna(value):

                continue

            value = str(value)

            if value.strip() == "None above":

                data.at[idx, "Symptom_None"] = 1

                continue

            tokens = [

                x.strip()

                for x in value.split(";")

            ]

            count = 0

            for token in tokens:

                if token in cls.SYMPTOMS:

                    feature = cls.SYMPTOMS[token]

                    data.at[idx, feature] = 1

                    count += 1

            data.at[idx, "SymptomCount"] = count

        return data

    @staticmethod
    def validate(df):

        symptom_columns = [

            "Symptom_IrregularCycle",

            "Symptom_WeightGain",

            "Symptom_Acne",

            "Symptom_Hirsutism",

            "Symptom_None"

        ]

        for col in symptom_columns:

            values = set(df[col].unique())

            if not values.issubset({0, 1}):

                raise ValueError(

                    f"{col} is not binary."

                )

        return True