# ============================================================
# Phase 11-B3 v2 - Corrected Refinement of Reliability-Aware Scenario Results
# ============================================================
#
# Fixes relative to Phase 11-B3 v1:
#   1) Prevents accidental duplication caused by merging biochar and bio-oil
#      scope rows using only group_type/group_value.
#   2) Filters scope metadata to target == "biochar" before merging.
#   3) Adds duplicate checks after each major merge.
#   4) Produces v2 output files so original files are preserved.
#
# Important:
# These outputs are still reliability-aware screening windows, not validated
# optimum conditions.
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path


WINDOWS_PATH = Path("phase11b2_reliable_operating_windows.xlsx")
COMPARISON_PATH = Path("phase11b2_yield_vs_reliable_comparison.xlsx")
AD_DIST_PATH = Path("phase11b2_candidate_ad_distribution.xlsx")
TOP_SCENARIOS_PATH = Path("phase11b2_top_ranked_scenarios.xlsx")
SCOPE_PATH = Path("phase11_recommended_scope.xlsx")

required_paths = [WINDOWS_PATH, COMPARISON_PATH, AD_DIST_PATH, SCOPE_PATH]

for path in required_paths:
    assert path.exists(), f"File not found: {path.resolve()}"

windows = pd.read_excel(WINDOWS_PATH)
comparison = pd.read_excel(COMPARISON_PATH)
ad_dist = pd.read_excel(AD_DIST_PATH)
scope = pd.read_excel(SCOPE_PATH)
top_scenarios = pd.read_excel(TOP_SCENARIOS_PATH) if TOP_SCENARIOS_PATH.exists() else None

print("Loaded:", WINDOWS_PATH, windows.shape)
print("Loaded:", COMPARISON_PATH, comparison.shape)
print("Loaded:", AD_DIST_PATH, ad_dist.shape)
print("Loaded:", SCOPE_PATH, scope.shape)

if top_scenarios is not None:
    print("Loaded:", TOP_SCENARIOS_PATH, top_scenarios.shape)


CRITICAL_WINDOW_COLUMNS = [
    "temperature_c_median",
    "residence_time_min_median",
]

MAIN_FAMILY_MIN_ACCEPTABLE_FRACTION = 0.60
CASE_STUDY_MIN_ACCEPTABLE_FRACTION = 0.70

MAX_OUT_OF_DOMAIN_FRACTION_MAIN = 0.40
MAX_OUT_OF_DOMAIN_FRACTION_CASE = 0.30

YIELD_ONLY_HIGH_RISK_OOD_FRACTION = 0.50


def safe_round_numeric(df, decimals=3):
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(decimals)
    return out


def assert_no_duplicate_groups(df, label):
    if not {"group_type", "group_value"}.issubset(df.columns):
        return

    duplicate_count = df[["group_type", "group_value"]].duplicated().sum()

    if duplicate_count > 0:
        duplicates = df[df[["group_type", "group_value"]].duplicated(keep=False)][
            ["group_type", "group_value"]
        ].drop_duplicates()

        print(f"\nWARNING: duplicate groups detected in {label}: {duplicate_count}")
        print(duplicates.to_string(index=False))


def get_group_ad_distribution(ad_dist_df):
    pivot = (
        ad_dist_df
        .pivot_table(
            index=["group_type", "group_value"],
            columns="combined_ad_class",
            values="fraction",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for col in ["in_domain", "near_domain", "out_of_domain", "unknown"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot = pivot.rename(columns={
        "in_domain": "candidate_fraction_in_domain",
        "near_domain": "candidate_fraction_near_domain",
        "out_of_domain": "candidate_fraction_out_of_domain",
        "unknown": "candidate_fraction_unknown",
    })

    pivot["candidate_fraction_in_or_near_domain"] = (
        pivot["candidate_fraction_in_domain"]
        + pivot["candidate_fraction_near_domain"]
    )

    assert_no_duplicate_groups(pivot, "ad_group_distribution")

    return pivot


def extract_ranking_type_summary(comparison_df, ranking_type):
    subset = comparison_df[comparison_df["ranking_type"] == ranking_type].copy()

    rename_map = {}
    for col in subset.columns:
        if col not in ["group_type", "group_value", "ranking_type"]:
            rename_map[col] = f"{ranking_type}_{col}"

    subset = subset.rename(columns=rename_map)
    subset = subset.drop(columns=["ranking_type"])

    assert_no_duplicate_groups(subset, f"comparison_{ranking_type}")

    return subset


def add_biochar_scope_info(table, scope_df):
    """
    Corrected merge:
    The previous v1 script merged scope metadata only on group_type/group_value.
    Because scope contains both biochar and bio-oil rows, that created duplicate
    rows and mixed bio-oil evidence counts into biochar scenario tables.

    This version filters scope to target == biochar before merging.
    """
    needed_cols = [
        "target",
        "group_type",
        "group_value",
        "n_rows",
        "n_sources",
        "recommended_role",
        "claim_strength",
    ]

    existing = [col for col in needed_cols if col in scope_df.columns]

    if not {"target", "group_type", "group_value"}.issubset(existing):
        raise ValueError("Scope file must contain target, group_type, and group_value.")

    scope_small = scope_df[existing].copy()

    scope_small = scope_small[scope_small["target"] == "biochar"].copy()

    scope_small = scope_small.drop_duplicates(
        subset=["target", "group_type", "group_value"],
        keep="first",
    )

    scope_small = scope_small.rename(columns={
        "n_rows": "evidence_n_rows",
        "n_sources": "evidence_n_sources",
        "recommended_role": "scope_recommended_role",
        "claim_strength": "scope_claim_strength",
    })

    # Drop target before merge because all retained rows are biochar.
    scope_small = scope_small.drop(columns=["target"])

    assert_no_duplicate_groups(scope_small, "biochar_scope_small")

    out = table.merge(
        scope_small,
        on=["group_type", "group_value"],
        how="left",
        validate="one_to_one",
    )

    return out


def has_missing_critical_values(row):
    missing_cols = []

    for col in CRITICAL_WINDOW_COLUMNS:
        if col not in row.index or pd.isna(row[col]):
            missing_cols.append(col)

    return missing_cols


def classify_group(row):
    group_type = row["group_type"]

    missing_critical = has_missing_critical_values(row)

    if missing_critical:
        return "exclude_from_operating_windows", "missing_critical_window_values:" + ";".join(missing_critical)

    frac_acceptable = row.get("candidate_fraction_in_or_near_domain", np.nan)
    frac_ood = row.get("candidate_fraction_out_of_domain", np.nan)

    if group_type == "feedstock_family":
        if frac_acceptable >= MAIN_FAMILY_MIN_ACCEPTABLE_FRACTION and frac_ood <= MAX_OUT_OF_DOMAIN_FRACTION_MAIN:
            return "main_text_table", "family_group_with_sufficient_ad_support"
        if frac_acceptable >= 0.50:
            return "supplementary_table", "family_group_with_marginal_ad_support"
        return "warning_only", "family_group_with_weak_ad_support"

    if group_type == "feedstock":
        if frac_acceptable >= CASE_STUDY_MIN_ACCEPTABLE_FRACTION and frac_ood <= MAX_OUT_OF_DOMAIN_FRACTION_CASE:
            return "case_study_table", "feedstock_case_with_sufficient_ad_support"
        if frac_acceptable >= 0.50:
            return "supplementary_table", "feedstock_case_with_marginal_ad_support"
        return "warning_only", "feedstock_case_with_weak_ad_support"

    return "exclude_from_operating_windows", "unknown_group_type"


# ------------------------------------------------------------
# Build corrected group-level table
# ------------------------------------------------------------

assert_no_duplicate_groups(windows, "input_windows")

ad_group = get_group_ad_distribution(ad_dist)

yield_only_summary = extract_ranking_type_summary(
    comparison,
    ranking_type="yield_only",
)

reliable_summary = extract_ranking_type_summary(
    comparison,
    ranking_type="reliability_aware_lcb90_ad_filtered",
)

group_table = windows.merge(
    ad_group,
    on=["group_type", "group_value"],
    how="left",
    validate="one_to_one",
)

group_table = group_table.merge(
    yield_only_summary,
    on=["group_type", "group_value"],
    how="left",
    validate="one_to_one",
)

group_table = group_table.merge(
    reliable_summary,
    on=["group_type", "group_value"],
    how="left",
    validate="one_to_one",
)

group_table = add_biochar_scope_info(group_table, scope)

assert_no_duplicate_groups(group_table, "corrected_group_table")

classes = group_table.apply(classify_group, axis=1)
group_table["reporting_class"] = [x[0] for x in classes]
group_table["reporting_reason"] = [x[1] for x in classes]

if "yield_only_fraction_out_of_domain" in group_table.columns and "reliability_aware_lcb90_ad_filtered_fraction_out_of_domain" in group_table.columns:
    group_table["yield_only_minus_reliable_ood_fraction"] = (
        group_table["yield_only_fraction_out_of_domain"]
        - group_table["reliability_aware_lcb90_ad_filtered_fraction_out_of_domain"]
    )
else:
    group_table["yield_only_minus_reliable_ood_fraction"] = np.nan

if "yield_only_fraction_out_of_domain" in group_table.columns:
    group_table["yield_only_high_risk_flag"] = (
        group_table["yield_only_fraction_out_of_domain"] >= YIELD_ONLY_HIGH_RISK_OOD_FRACTION
    )
else:
    group_table["yield_only_high_risk_flag"] = np.nan


main_family_windows = group_table[
    (group_table["group_type"] == "feedstock_family")
    & (group_table["reporting_class"] == "main_text_table")
].copy()

case_study_windows = group_table[
    (group_table["group_type"] == "feedstock")
    & (group_table["reporting_class"] == "case_study_table")
].copy()

supplementary_windows = group_table[
    group_table["reporting_class"] == "supplementary_table"
].copy()

excluded_or_warning = group_table[
    group_table["reporting_class"].isin([
        "warning_only",
        "exclude_from_operating_windows",
    ])
].copy()


yield_only_risk_cols = [
    "group_type",
    "group_value",
    "candidate_fraction_in_or_near_domain",
    "candidate_fraction_out_of_domain",
    "yield_only_mean_predicted_biochar_yield",
    "yield_only_mean_lower_bound_90",
    "yield_only_fraction_in_domain",
    "yield_only_fraction_near_domain",
    "yield_only_fraction_out_of_domain",
    "reliability_aware_lcb90_ad_filtered_mean_predicted_biochar_yield",
    "reliability_aware_lcb90_ad_filtered_mean_lower_bound_90",
    "reliability_aware_lcb90_ad_filtered_fraction_in_domain",
    "reliability_aware_lcb90_ad_filtered_fraction_near_domain",
    "reliability_aware_lcb90_ad_filtered_fraction_out_of_domain",
    "yield_only_minus_reliable_ood_fraction",
    "yield_only_high_risk_flag",
    "reporting_class",
    "reporting_reason",
]

yield_only_risk_cols = [col for col in yield_only_risk_cols if col in group_table.columns]

yield_only_risk_table = group_table[yield_only_risk_cols].copy()

yield_only_risk_table = yield_only_risk_table.sort_values(
    ["yield_only_high_risk_flag", "yield_only_fraction_out_of_domain"],
    ascending=[False, False],
)


window_cols = [
    "group_type",
    "group_value",
    "evidence_n_rows",
    "evidence_n_sources",
    "n_top_scenarios",
    "mean_predicted_biochar_yield",
    "median_predicted_biochar_yield",
    "mean_lower_bound_90",
    "median_lower_bound_90",
    "mean_ensemble_std",
    "candidate_fraction_in_or_near_domain",
    "candidate_fraction_out_of_domain",
    "fraction_in_domain",
    "fraction_near_domain",
    "fraction_out_of_domain",
    "temperature_c_p10",
    "temperature_c_median",
    "temperature_c_p90",
    "residence_time_min_p10",
    "residence_time_min_median",
    "residence_time_min_p90",
    "heating_rate_c_min_p10",
    "heating_rate_c_min_median",
    "heating_rate_c_min_p90",
    "gas_flow_rate_l_min_p10",
    "gas_flow_rate_l_min_median",
    "gas_flow_rate_l_min_p90",
    "particle_size_mm_p10",
    "particle_size_mm_median",
    "particle_size_mm_p90",
    "raw_material_supply_g_p10",
    "raw_material_supply_g_median",
    "raw_material_supply_g_p90",
    "reporting_class",
    "reporting_reason",
]

window_cols = [col for col in window_cols if col in group_table.columns]

main_family_windows_compact = main_family_windows[window_cols].copy()
case_study_windows_compact = case_study_windows[window_cols].copy()
supplementary_windows_compact = supplementary_windows[window_cols].copy()
excluded_or_warning_compact = excluded_or_warning[
    [col for col in window_cols if col in excluded_or_warning.columns]
].copy()


top_scenario_examples = None

if top_scenarios is not None:
    keep_groups = pd.concat([
        main_family_windows[["group_type", "group_value"]],
        case_study_windows[["group_type", "group_value"]],
    ], ignore_index=True)

    if len(keep_groups) > 0:
        top_scenarios = top_scenarios.merge(
            keep_groups.assign(keep_for_main_or_case=True),
            on=["group_type", "group_value"],
            how="left",
        )

        top_scenario_examples = top_scenarios[
            (top_scenarios["keep_for_main_or_case"] == True)
            & (top_scenarios["ranking_type"] == "reliability_aware_lcb90_ad_filtered")
        ].copy()

        top_scenario_examples = (
            top_scenario_examples
            .sort_values(
                ["group_type", "group_value", "reliable_score_lcb90"],
                ascending=[True, True, False]
            )
            .groupby(["group_type", "group_value"], as_index=False)
            .head(5)
        )


interpretation_notes = pd.DataFrame([
    {
        "item": "main_claim",
        "recommended_wording": (
            "Reliability-aware ranking reduced the dominance of out-of-domain "
            "yield-only candidates and produced more conservative, empirically "
            "supported biochar scenario windows."
        ),
        "avoid_wording": "The true optimum pyrolysis conditions were determined.",
    },
    {
        "item": "biochar_scope",
        "recommended_wording": (
            "The operating windows should be interpreted as pre-experimental "
            "screening targets for biochar yield, not as experimentally validated "
            "or industrially deployable set-points."
        ),
        "avoid_wording": "These operating conditions can be directly recommended for industrial operation.",
    },
    {
        "item": "yield_only_ranking",
        "recommended_wording": (
            "Yield-only ranking is retained as a diagnostic comparison because it "
            "often selects candidates with weak applicability-domain support."
        ),
        "avoid_wording": "Yield-only top candidates are optimal.",
    },
    {
        "item": "uncertainty",
        "recommended_wording": (
            "The lower-bound score is based on an empirical conformal interval; "
            "because the absolute conformal half-width is constant, applicability-domain "
            "filtering is the main driver of differences between yield-only and "
            "reliability-aware candidate lists."
        ),
        "avoid_wording": "The model provides complete pointwise process uncertainty.",
    },
])


group_table.to_excel("phase11b3_v2_refined_group_table.xlsx", index=False)
main_family_windows_compact.to_excel("phase11b3_v2_main_family_windows.xlsx", index=False)
case_study_windows_compact.to_excel("phase11b3_v2_case_study_windows.xlsx", index=False)
supplementary_windows_compact.to_excel("phase11b3_v2_supplementary_windows.xlsx", index=False)
excluded_or_warning_compact.to_excel("phase11b3_v2_excluded_or_warning_groups.xlsx", index=False)
yield_only_risk_table.to_excel("phase11b3_v2_yield_only_risk_table.xlsx", index=False)
interpretation_notes.to_excel("phase11b3_v2_interpretation_notes.xlsx", index=False)

if top_scenario_examples is not None:
    top_scenario_examples.to_excel("phase11b3_v2_top_scenario_examples.xlsx", index=False)

print("\nSaved Phase 11-B3 v2 files:")
print("- phase11b3_v2_refined_group_table.xlsx")
print("- phase11b3_v2_main_family_windows.xlsx")
print("- phase11b3_v2_case_study_windows.xlsx")
print("- phase11b3_v2_supplementary_windows.xlsx")
print("- phase11b3_v2_excluded_or_warning_groups.xlsx")
print("- phase11b3_v2_yield_only_risk_table.xlsx")
print("- phase11b3_v2_interpretation_notes.xlsx")

if top_scenario_examples is not None:
    print("- phase11b3_v2_top_scenario_examples.xlsx")

print("\nReporting-class counts:")
print(
    group_table
    .groupby(["group_type", "reporting_class"], as_index=False)
    .agg(n_groups=("group_value", "nunique"))
    .to_string(index=False)
)

print("\nMain family windows:")
print(safe_round_numeric(main_family_windows_compact).to_string(index=False))

print("\nCase-study windows:")
print(safe_round_numeric(case_study_windows_compact).to_string(index=False))

print("\nImportant:")
print("Use v2 outputs. The v1 Phase 11-B3 outputs duplicated rows because scope metadata was merged without filtering to biochar.")
