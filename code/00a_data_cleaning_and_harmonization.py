# %%
import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path("SI_Data.xlsx")
REF_PATH = Path("SI_Data_References.xlsx")

print("Working directory:", Path.cwd())
print("SI_Data exists:", RAW_PATH.exists())
print("References exists:", REF_PATH.exists())

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Cannot find {RAW_PATH.resolve()}")

if not REF_PATH.exists():
    raise FileNotFoundError(f"Cannot find {REF_PATH.resolve()}")

COLUMN_MAP = {
    "Feed Stock Type": "feedstock",
    "Carbon": "carbon",
    "Hydrogen": "hydrogen",
    "Nitrogen": "nitrogen",
    "Oxygen": "oxygen",
    "Raw Material Supply (g)": "raw_material_supply_g",
    "Temperature (C )": "temperature_c",
    "Residence Time (min)": "residence_time_min",
    "Gas Flow Rate (L/min)": "gas_flow_rate_l_min",
    "Heating Rate (C/min)": "heating_rate_c_min",
    "Moisture (%)": "moisture",
    "VM": "volatile_matter",
    "Ash": "ash",
    "FC": "fixed_carbon",
    "Particle Size (mm)": "particle_size_mm",
    "Biochar Yield (%)": "biochar_yield",
    "Bio Oil Yield (%)": "bio_oil_yield",
    "References 1": "reference_1",
    "References 2": "reference_2",
}

NUMERIC_COLUMNS = [
    "carbon",
    "hydrogen",
    "nitrogen",
    "oxygen",
    "raw_material_supply_g",
    "temperature_c",
    "residence_time_min",
    "gas_flow_rate_l_min",
    "heating_rate_c_min",
    "moisture",
    "volatile_matter",
    "ash",
    "fixed_carbon",
    "particle_size_mm",
    "biochar_yield",
    "bio_oil_yield",
]

INPUT_COLUMNS = [
    "carbon",
    "hydrogen",
    "nitrogen",
    "oxygen",
    "raw_material_supply_g",
    "temperature_c",
    "residence_time_min",
    "gas_flow_rate_l_min",
    "heating_rate_c_min",
    "moisture",
    "volatile_matter",
    "ash",
    "fixed_carbon",
    "particle_size_mm",
]


def clean_numeric_cell(x):
    """
    Conservative conversion of text-formatted numeric values.
    Ambiguous values are returned as NaN and should be flagged.
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    s = s.replace("\n", "")
    s = s.replace(" ", "")

    # Fix repeated decimal points only when pattern is simple, e.g. 53..5 -> 53.5
    if re.fullmatch(r"\d+\.\.+\d+", s):
        s = re.sub(r"\.+", ".", s)

    # Ambiguous case: more than one decimal point remains
    if s.count(".") > 1:
        return np.nan

    try:
        return float(s)
    except ValueError:
        return np.nan


def flag_numeric_issues(original_series, cleaned_series):
    flags = pd.Series(False, index=original_series.index)

    for idx, value in original_series.items():
        if pd.isna(value):
            continue

        raw = str(value).strip()

        # Flag suspicious values that required non-trivial handling
        if "\n" in raw or " " in raw or ".." in raw or raw.count(".") > 1:
            flags.loc[idx] = True

        # If original was non-empty but conversion failed
        if pd.isna(cleaned_series.loc[idx]):
            flags.loc[idx] = True

    return flags


def load_and_clean_raw_data(path):
    raw = pd.read_excel(path, sheet_name="Sheet1")
    df = raw.copy(deep=True)

    df = df.rename(columns=COLUMN_MAP)

    # Basic feedstock harmonization
    df["feedstock_raw"] = df["feedstock"]
    df["feedstock"] = df["feedstock"].astype(str).str.strip()

    feedstock_replacements = {
        "Bamboo ": "Bamboo",
        "Switch Grass": "Switchgrass",
    }
    df["feedstock"] = df["feedstock"].replace(feedstock_replacements)

    # Numeric conversion with flags
    for col in NUMERIC_COLUMNS:
        original = df[col].copy()
        df[col] = original.apply(clean_numeric_cell)
        df[f"{col}_conversion_flag"] = flag_numeric_issues(original, df[col])

    # Reference columns as nullable integers
    for col in ["reference_1", "reference_2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Source group reconstruction
    df["source_group"] = df["reference_1"].copy()
    missing_ref1 = df["source_group"].isna()
    df.loc[missing_ref1, "source_group"] = df.loc[missing_ref1, "reference_2"]

    df["source_confidence"] = np.where(
        df["reference_1"].notna(),
        "primary_reference_1",
        "reconstructed_from_reference_2"
    )

    # Conservative source-pair group for sensitivity analysis
    df["source_pair_group"] = (
        df["reference_1"].astype("string").fillna("NA") + "_" +
        df["reference_2"].astype("string").fillna("NA")
    )

    # Composition audit columns
    df["ultimate_sum"] = df[["carbon", "hydrogen", "nitrogen", "oxygen"]].sum(axis=1, min_count=4)
    df["proximate_sum"] = df[["moisture", "volatile_matter", "ash", "fixed_carbon"]].sum(axis=1, min_count=4)

    df["ultimate_outside_80_110"] = ~df["ultimate_sum"].between(80, 110)
    df["proximate_outside_90_110"] = ~df["proximate_sum"].between(90, 110)

    # Duplicate flags
    df["full_duplicate_flag"] = df.duplicated(keep=False)

    input_duplicate_cols = INPUT_COLUMNS + ["feedstock"]
    df["duplicate_input_flag"] = df.duplicated(subset=input_duplicate_cols, keep=False)

    # Target availability
    df["has_biochar"] = df["biochar_yield"].notna()
    df["has_bio_oil"] = df["bio_oil_yield"].notna()
    df["has_both_targets"] = df["has_biochar"] & df["has_bio_oil"]

    return raw, df


raw_data, clean_audit_data = load_and_clean_raw_data(RAW_PATH)

biochar_dataset = clean_audit_data[clean_audit_data["has_biochar"]].copy()
bio_oil_dataset = clean_audit_data[clean_audit_data["has_bio_oil"]].copy()
paired_dataset = clean_audit_data[clean_audit_data["has_both_targets"]].copy()

print("Raw shape:", raw_data.shape)
print("Clean audit shape:", clean_audit_data.shape)
print("Biochar dataset:", biochar_dataset.shape)
print("Bio-oil dataset:", bio_oil_dataset.shape)
print("Paired dataset:", paired_dataset.shape)
print("Unique feedstocks:", clean_audit_data["feedstock"].nunique())
print("Unique source groups:", clean_audit_data["source_group"].nunique())

# %%
clean_audit_data.to_excel("clean_audit_data.xlsx", index=False)
biochar_dataset.to_excel("biochar_dataset.xlsx", index=False)
bio_oil_dataset.to_excel("bio_oil_dataset.xlsx", index=False)
paired_dataset.to_excel("paired_dataset.xlsx", index=False)

print("Cleaned datasets saved successfully.")
