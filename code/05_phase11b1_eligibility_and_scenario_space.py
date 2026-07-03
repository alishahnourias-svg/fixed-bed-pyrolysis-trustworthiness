# ============================================================
# Phase 11-B1 - Eligibility Audit and Safe Scenario-Space Construction
# Project:
# From Accuracy to Trustworthiness:
# Grouped Validation, Applicability-Domain Mapping,
# and Reliability-Aware Optimization for ML-Based
# Fixed-Bed Biomass Pyrolysis Yield Prediction
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# 0) General settings
# ------------------------------------------------------------

DATA_PATH = Path("clean_audit_data_with_family.xlsx")
assert DATA_PATH.exists(), f"File not found: {DATA_PATH.resolve()}"

df = pd.read_excel(DATA_PATH)
df = df.reset_index(drop=True)
df["row_id"] = df.index.astype(int)
df["excel_row_number"] = df.index.astype(int) + 2

print("Loaded:", DATA_PATH)
print("Shape:", df.shape)


# ------------------------------------------------------------
# 1) Column definitions
# ------------------------------------------------------------

TARGETS = {
    "biochar": "biochar_yield",
    "bio_oil": "bio_oil_yield",
}

COMPOSITION_FEATURES = [
    "carbon",
    "hydrogen",
    "nitrogen",
    "oxygen",
    "moisture",
    "volatile_matter",
    "fixed_carbon",
    "ash",
]

OPERATIONAL_FEATURES = [
    "particle_size_mm",
    "temperature_c",
    "heating_rate_c_min",
    "gas_flow_rate_l_min",
    "residence_time_min",
    "raw_material_supply_g",
]

FEATURES_FOR_SCENARIO = COMPOSITION_FEATURES + OPERATIONAL_FEATURES

GROUP_COLUMNS = [
    "feedstock",
    "feedstock_family",
]

REFERENCE_COLUMNS = [
    "source_group",
    "source_pair_group",
]


# ------------------------------------------------------------
# 2) Conservative eligibility thresholds
# ------------------------------------------------------------
# These thresholds are deliberately conservative. They are not universal rules;
# they are used here to prevent fragile optimization on small or single-source groups.

ELIGIBILITY_RULES = {
    "feedstock": {
        "min_rows": 30,
        "min_sources": 3,
        "min_nonmissing_temperature": 20,
        "min_nonmissing_residence_time": 10,
    },
    "feedstock_family": {
        "min_rows": 100,
        "min_sources": 5,
        "min_nonmissing_temperature": 60,
        "min_nonmissing_residence_time": 30,
    },
}


# ------------------------------------------------------------
# 3) Required-column check
# ------------------------------------------------------------

required_cols = (
    list(TARGETS.values())
    + COMPOSITION_FEATURES
    + OPERATIONAL_FEATURES
    + GROUP_COLUMNS
    + REFERENCE_COLUMNS
)

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("Required-column check passed.")


# ------------------------------------------------------------
# 4) Utility functions
# ------------------------------------------------------------

def safe_group_string(series):
    return (
        series.astype("string")
        .fillna("MISSING")
        .str.replace(r"\.0$", "", regex=True)
    )


for col in GROUP_COLUMNS + REFERENCE_COLUMNS:
    df[col] = safe_group_string(df[col])


def quantile_or_nan(series, q):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.quantile(q))


def summarize_group(data, target_key, group_col):
    target_col = TARGETS[target_key]
    rules = ELIGIBILITY_RULES[group_col]

    rows = []

    usable = data[data[target_col].notna()].copy()

    for group_value, g in usable.groupby(group_col):
        row = {
            "target": target_key,
            "group_type": group_col,
            "group_value": group_value,
            "n_rows": len(g),
            "n_sources": g["source_group"].nunique(),
            "n_source_pairs": g["source_pair_group"].nunique(),
            "target_mean": g[target_col].mean(),
            "target_median": g[target_col].median(),
            "target_min": g[target_col].min(),
            "target_max": g[target_col].max(),
        }

        for col in COMPOSITION_FEATURES + OPERATIONAL_FEATURES:
            row[f"{col}_nonmissing"] = g[col].notna().sum()
            row[f"{col}_q05"] = quantile_or_nan(g[col], 0.05)
            row[f"{col}_q10"] = quantile_or_nan(g[col], 0.10)
            row[f"{col}_median"] = quantile_or_nan(g[col], 0.50)
            row[f"{col}_q90"] = quantile_or_nan(g[col], 0.90)
            row[f"{col}_q95"] = quantile_or_nan(g[col], 0.95)
            row[f"{col}_min"] = pd.to_numeric(g[col], errors="coerce").min()
            row[f"{col}_max"] = pd.to_numeric(g[col], errors="coerce").max()

        row["eligible_by_rows"] = row["n_rows"] >= rules["min_rows"]
        row["eligible_by_sources"] = row["n_sources"] >= rules["min_sources"]
        row["eligible_by_temperature"] = row["temperature_c_nonmissing"] >= rules["min_nonmissing_temperature"]
        row["eligible_by_residence_time"] = row["residence_time_min_nonmissing"] >= rules["min_nonmissing_residence_time"]

        row["eligible_for_scenario_analysis"] = all([
            row["eligible_by_rows"],
            row["eligible_by_sources"],
            row["eligible_by_temperature"],
            row["eligible_by_residence_time"],
        ])

        failed = []
        if not row["eligible_by_rows"]:
            failed.append("too_few_rows")
        if not row["eligible_by_sources"]:
            failed.append("too_few_sources")
        if not row["eligible_by_temperature"]:
            failed.append("too_few_temperature_values")
        if not row["eligible_by_residence_time"]:
            failed.append("too_few_residence_time_values")

        row["exclusion_reason"] = ";".join(failed) if failed else ""

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 5) Eligibility audit
# ------------------------------------------------------------

eligibility_tables = []

for target_key in TARGETS:
    for group_col in GROUP_COLUMNS:
        eligibility_tables.append(
            summarize_group(df, target_key=target_key, group_col=group_col)
        )

phase11_eligibility_audit = pd.concat(eligibility_tables, ignore_index=True)

phase11_eligible_entities = phase11_eligibility_audit[
    phase11_eligibility_audit["eligible_for_scenario_analysis"]
].copy()

phase11_ineligible_entities = phase11_eligibility_audit[
    ~phase11_eligibility_audit["eligible_for_scenario_analysis"]
].copy()


# ------------------------------------------------------------
# 6) Build conservative scenario-space definitions
# ------------------------------------------------------------
# This does NOT optimize yet. It defines safe empirical ranges for later
# candidate generation and ranking.
#
# Operational variables are restricted to the empirical q10-q90 range
# inside the same feedstock or feedstock family.
#
# Composition variables are not freely optimized. They are kept from
# real observed rows in the next phase.

scenario_rows = []

for _, row in phase11_eligible_entities.iterrows():
    target_key = row["target"]
    group_col = row["group_type"]
    group_value = row["group_value"]

    subset = df[
        (df[TARGETS[target_key]].notna())
        & (df[group_col] == group_value)
    ].copy()

    scenario = {
        "target": target_key,
        "group_type": group_col,
        "group_value": group_value,
        "n_rows": len(subset),
        "n_sources": subset["source_group"].nunique(),
        "n_source_pairs": subset["source_pair_group"].nunique(),
        "scenario_policy": (
            "Composition variables must be taken from real observed rows. "
            "Operational variables may be sampled only within empirical q10-q90 "
            "ranges of the same group. Candidate scenarios must later be checked "
            "against applicability-domain and uncertainty filters."
        ),
    }

    for col in OPERATIONAL_FEATURES:
        scenario[f"{col}_lower_q10"] = quantile_or_nan(subset[col], 0.10)
        scenario[f"{col}_upper_q90"] = quantile_or_nan(subset[col], 0.90)
        scenario[f"{col}_observed_min"] = pd.to_numeric(subset[col], errors="coerce").min()
        scenario[f"{col}_observed_max"] = pd.to_numeric(subset[col], errors="coerce").max()
        scenario[f"{col}_nonmissing"] = subset[col].notna().sum()

    for col in COMPOSITION_FEATURES:
        scenario[f"{col}_observed_q10"] = quantile_or_nan(subset[col], 0.10)
        scenario[f"{col}_observed_median"] = quantile_or_nan(subset[col], 0.50)
        scenario[f"{col}_observed_q90"] = quantile_or_nan(subset[col], 0.90)

    scenario_rows.append(scenario)

phase11_scenario_space_definitions = pd.DataFrame(scenario_rows)


# ------------------------------------------------------------
# 7) Determine recommended optimization scope
# ------------------------------------------------------------
# Biochar is the main optimization target.
# Bio-oil is retained only for warning/sensitivity, not strong recommendation.

scope_rows = []

for _, row in phase11_eligible_entities.iterrows():
    target_key = row["target"]
    group_col = row["group_type"]
    group_value = row["group_value"]

    if target_key == "biochar":
        recommended_role = "main_reliability_aware_scenario_ranking"
        claim_strength = "moderate_screening"
        recommended_next_action = (
            "Use for candidate scenario generation and reliability-aware ranking. "
            "Report windows, not exact optima."
        )
    else:
        recommended_role = "warning_only_or_sensitivity"
        claim_strength = "limited_warning"
        recommended_next_action = (
            "Use only to demonstrate risk of yield-only optimization. "
            "Do not report as operational recommendation."
        )

    scope_rows.append({
        "target": target_key,
        "group_type": group_col,
        "group_value": group_value,
        "n_rows": row["n_rows"],
        "n_sources": row["n_sources"],
        "recommended_role": recommended_role,
        "claim_strength": claim_strength,
        "recommended_next_action": recommended_next_action,
    })

phase11_recommended_scope = pd.DataFrame(scope_rows)


# ------------------------------------------------------------
# 8) Compact counts for console
# ------------------------------------------------------------

scope_counts = (
    phase11_eligibility_audit
    .groupby(["target", "group_type", "eligible_for_scenario_analysis"], as_index=False)
    .agg(n_groups=("group_value", "nunique"))
)

biochar_scope = phase11_recommended_scope[
    phase11_recommended_scope["target"] == "biochar"
].copy()

bio_oil_scope = phase11_recommended_scope[
    phase11_recommended_scope["target"] == "bio_oil"
].copy()


# ------------------------------------------------------------
# 9) Save outputs
# ------------------------------------------------------------

phase11_eligibility_audit.to_excel("phase11_eligibility_audit.xlsx", index=False)
phase11_eligible_entities.to_excel("phase11_eligible_entities.xlsx", index=False)
phase11_ineligible_entities.to_excel("phase11_ineligible_entities.xlsx", index=False)
phase11_scenario_space_definitions.to_excel("phase11_scenario_space_definitions.xlsx", index=False)
phase11_recommended_scope.to_excel("phase11_recommended_scope.xlsx", index=False)
scope_counts.to_excel("phase11_scope_counts.xlsx", index=False)

print("\nSaved Phase 11-B1 files:")
print("- phase11_eligibility_audit.xlsx")
print("- phase11_eligible_entities.xlsx")
print("- phase11_ineligible_entities.xlsx")
print("- phase11_scenario_space_definitions.xlsx")
print("- phase11_recommended_scope.xlsx")
print("- phase11_scope_counts.xlsx")


# ------------------------------------------------------------
# 10) Console preview
# ------------------------------------------------------------

print("\nEligibility counts:")
print(scope_counts.to_string(index=False))

print("\nRecommended biochar scope:")
if len(biochar_scope) == 0:
    print("No eligible biochar groups found under current conservative thresholds.")
else:
    print(
        biochar_scope
        .sort_values(["group_type", "n_rows"], ascending=[True, False])
        [["group_type", "group_value", "n_rows", "n_sources", "recommended_role", "claim_strength"]]
        .head(30)
        .to_string(index=False)
    )

print("\nBio-oil groups retained only for warning/sensitivity:")
if len(bio_oil_scope) == 0:
    print("No eligible bio-oil groups found under current conservative thresholds.")
else:
    print(
        bio_oil_scope
        .sort_values(["group_type", "n_rows"], ascending=[True, False])
        [["group_type", "group_value", "n_rows", "n_sources", "recommended_role", "claim_strength"]]
        .head(30)
        .to_string(index=False)
    )

print("\nImportant:")
print("This script does not generate optimum conditions.")
print("It only defines which feedstocks/families have enough evidence for scenario ranking.")
print("The next script should generate candidate scenarios only for eligible groups.")
