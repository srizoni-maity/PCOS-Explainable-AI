from pathlib import Path
import json
import pandas as pd

from src.core.loader import load_raw_dataset
from src.core.validator import validate_column_values


def standardize_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "", regex=False)
    )
    return df


def normalize_text_columns(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        df[col] = df[col].replace({
            "nan": pd.NA,
            "None": pd.NA,
            "": pd.NA
        })

    return df


def remove_duplicates(df):
    df = df.dropna().reset_index(drop=True)
    before = len(df)
    after = len(df)
    print(f"Removed incomplete rows : {before - after}")
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, removed


def handle_username_column(df):
    username_cols = [c for c in df.columns if "user" in c.lower()]

    duplicate_count = 0

    if username_cols:
        col = username_cols[0]
        duplicate_count = df[col].duplicated().sum()

        # Remove identifier only
        df = df.drop(columns=[col])

    return df, duplicate_count


def create_validation_report(df, invalid_values):
    report = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "invalid_values": invalid_values,
        "class_distribution": df["PCOS"].value_counts(dropna=False).to_dict()
        if "PCOS" in df.columns else {}
    }
    return report


def save_audit_log(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==========================================================
# Canonical Column Names
# ==========================================================

COLUMN_MAPPING = {
    "1.AgeGroup": "AgeGroup",
    "2.AreyoucurrentlyacollegeuniversitystudentinWestBengal?": "Student",
    "3.Doyoulivein:": "Residence",
    "4.HaveyoueverbeendiagnosedwithPCOSPolycysticOvarySyndrome?": "PCOS",
    "5.DoyouexperienceanyofthefollowingsymptomscommonlyassociatedwithPCOS?": "Symptoms",
    "6.HowfamiliarareyouwithPCOSanditsriskfactors?": "PCOSAwareness",
    "7.Howoftendoyouconsumepackagedorprocessedfoods?": "PackagedFood",
    "8.Howoftendoyoueatfriedfastfoods?": "FastFood",
    "9.Howoftendoyouusuallycheckfoodlabelsfortransfatorhydrogenatedoilcontentbeforebuying?": "FoodLabel",
    "10.Inyouropinion,“industrialtransfatsareharmfultohealth”?": "TransFatAwareness",
    "11.Howoftendoyouengageinphysicalexercise?": "Exercise",
    "12.Onaverage,howmanyhoursofsleepdoyougetpernight?": "Sleep",
    "13.Howwouldyourateyourstresslevelduetoacademicorpersonalfactors?": "Stress",
    "14.Howoftendoyouconsumesugarydrinks?": "SugaryDrinks",
    "15.Haveyoueverexperiencedorbeendiagnosedwithanyofthefollowing?": "MedicalHistory",
}

EXPECTED_COLUMNS = [
    "AgeGroup",
    "Student",
    "Residence",
    "PCOS",
    "Symptoms",
    "PCOSAwareness",
    "PackagedFood",
    "FastFood",
    "FoodLabel",
    "TransFatAwareness",
    "Exercise",
    "Sleep",
    "Stress",
    "SugaryDrinks",
    "MedicalHistory",
]

CATEGORY_REPLACEMENTS = {

    "Fried fast foods": None,

    "Check trans fat labels": None,

    "Industrial trans fats harmful": None,

    "Physical exercise": None,

    "Sleep": None,

    "Stress": None,

    "Sugary drinks": None,

    "Health conditions": None,

    "Modarate": "Moderate"

}

def run_preprocessing():
    df, config = load_raw_dataset()

    audit = []
    audit.append("=== PCOS DATA CLEANING AUDIT ===")
    audit.append(f"Original shape: {df.shape}")

    # Standardize column names
    df = standardize_column_names(df)
    audit.append(f"Columns standardized: {list(df.columns)}")

    # Rename questionnaire columns to canonical names
    df.rename(
        columns=COLUMN_MAPPING,
        inplace=True
    )
    df.replace(
        CATEGORY_REPLACEMENTS,
        inplace=True
    )
    # Verify every expected column exists
    missing = [
        c for c in EXPECTED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Column renaming failed. Missing columns: {missing}"
        )
    audit.append(f"Canonical columns: {list(df.columns)}")

    # Normalize text
    df = normalize_text_columns(df)

    # Remove exact duplicate rows
    df, row_dups = remove_duplicates(df)
    audit.append(f"Duplicate rows removed: {row_dups}")

    # Remove duplicate usernames if present
    df, user_dups = handle_username_column(df)
    audit.append(f"Duplicate usernames detected: {user_dups}")

    # Validate categorical values
    invalid_values = validate_column_values(df)

    if invalid_values:
        audit.append("Invalid values detected:")
        for col, vals in invalid_values.items():
            audit.append(f"  {col}: {vals}")
    else:
        audit.append("No invalid categorical values found.")

    # Final shape
    audit.append(f"Final shape: {df.shape}")
    
    print("\nCanonical Columns")
    for col in df.columns:
        print(f" - {col}")

    # Save cleaned dataset
    processed_path = config["paths"]["processed_data"]
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    # Save validation report
    report = create_validation_report(df, invalid_values)

    report_path = config["paths"]["validation_report"]
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # Save audit log
    save_audit_log(config["paths"]["audit_log"], audit)

    print("=" * 70)
    print("PCOS DATA CLEANING & VALIDATION")
    print("=" * 70)
    print(f"Final dataset shape : {df.shape}")
    print(f"Cleaned dataset     : {processed_path}")
    print(f"Validation report   : {report_path}")
    print(f"Audit log           : {config['paths']['audit_log']}")
    print("=" * 70)

    return df


if __name__ == "__main__":
    run_preprocessing()

