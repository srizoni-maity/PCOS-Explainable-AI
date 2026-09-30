ALLOWED_VALUES = {
    "AgeGroup": {
        "17 - 19 years",
        "20 - 22 years",
        "23 - 25 years"
    },

    "Student": {"Yes", "No"},

    "Residence": {"Urban", "Semi-urban", "Rural"},

    "PCOS": {"Yes", "No"},

    "Symptoms": {
        "Irregular periods",
        "Weight gain",
        "Acne",
        "Excess facial hair",
        "No symptoms"
    },

    "PCOSAwareness": {
        "Very familiar",
        "Somewhat familiar",
        "Heard of it, but not sure",
        "Never heard of it"
    },

    "PackagedFood": {
        "Daily", "3-5 times/week", "1-2 times/week", "Rarely"
    },

    "FastFood": {
        "Daily", "3-5 times/week", "1-2 times/week", "Rarely"
    },

    "FoodLabel": {
        "Always", "Sometimes", "Rarely", "Never"
    },

    "TransFatAwareness": {
        "Yes", "No", "Not sure"
    },

    "Exercise": {
        "Regular", "Occasional", "Never"
    },

    "Sleep": {
        "Less than 5 hours",
        "5-6 hours",
        "7-8 hours",
        "More than 8 hours"
    },

    "Stress": {
        "Low", "Moderate", "High"
    },

    "SugaryDrinks": {
        "Daily", "3-5 times/week", "1-2 times/week", "Rarely"
    },

    "MedicalHistory": {
        "Yes", "No", "Not sure"
    }
}


def validate_column_values(df):
    """Return invalid values found in each column."""
    invalid = {}

    for column, allowed in ALLOWED_VALUES.items():
        if column not in df.columns:
            invalid[column] = ["COLUMN_MISSING"]
            continue

        values = set(df[column].dropna().astype(str).str.strip())
        bad = sorted(values - allowed)

        if bad:
            invalid[column] = bad

    return invalid