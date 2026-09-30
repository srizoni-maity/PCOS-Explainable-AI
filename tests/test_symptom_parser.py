import pandas as pd

from src.core.symptom_parser import SymptomParser

df = pd.DataFrame({

    "Symptoms":[

        "Weight gain",

        "Weight gain;Acne or oily skin",

        "Irregular menstrual cycle",

        "None above",

        "Weight gain;Acne or oily skin;Irregular menstrual cycle"

    ]

})

df = SymptomParser.transform(df)

SymptomParser.validate(df)

print(df)