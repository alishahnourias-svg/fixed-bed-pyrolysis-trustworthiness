# ============================================================
# Phase 8 - Model Selection and Robustness Diagnostics
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
# 0) File paths
# ------------------------------------------------------------

SUMMARY_PATH = Path("phase7_v2_summary_results.xlsx")
PREDICTION_PATH = Path("phase7_v2_prediction_results.xlsx")

assert SUMMARY_PATH.exists(), f"File not found: {SUMMARY_PATH.resolve()}"

summary = pd.read_excel(SUMMARY_PATH)

print("Loaded:", SUMMARY_PATH)
print("Summary shape:", summary.shape)


# ------------------------------------------------------------
# 1) Required-column check
# ------------------------------------------------------------

REQUIRED_SUMMARY_COLUMNS = [
    "target",
    "protocol",
    "model",
    "validation_regime",
    "mean_r2",
    "mean_rmse",
    "mean_mae",
    "mean_median_ae",
    "pooled_r2",
    "pooled_rmse",
    "pooled_mae",
    "pooled_median_ae",
    "mean_r2_drop_vs_random",
    "mean_rmse_increase_vs_random",
    "mean_mae_increase_vs_random",
    "pooled_r2_drop_vs_random",
    "pooled_rmse_increase_vs_random",
    "pooled_mae_increase_vs_random",
]

missing = [col for col in REQUIRED_SUMMARY_COLUMNS if col not in summary.columns]
if missing:
    raise ValueError(f"Missing columns in summary file: {missing}")

print("Required-column check passed.")


# ------------------------------------------------------------
# 2) Define validation categories
# ------------------------------------------------------------

RANDOM_REGIME = "random_kfold"

GROUPED_REGIMES = [
    "source_group_kfold",
    "feedstock_group_kfold",
    "family_group_kfold",
]

CORE_GROUPED_REGIMES = [
    "source_group_kfold",
    "feedstock_group_kfold",
]

SUPPLEMENTARY_GROUPED_REGIMES = [
    "family_group_kfold",
]

summary["is_random"] = summary["validation_regime"].eq(RANDOM_REGIME)
summary["is_grouped"] = summary["validation_regime"].isin(GROUPED_REGIMES)
summary["is_core_grouped"] = summary["validation_regime"].isin(CORE_GROUPED_REGIMES)


# ------------------------------------------------------------
# 3) Best models in each target and validation regime
# ------------------------------------------------------------

best_by_regime = (
    summary
    .sort_values(["target", "validation_regime", "mean_rmse"])
    .groupby(["target", "validation_regime"], as_index=False)
    .head(5)
    .copy()
)

best_by_regime = best_by_regime[
    [
        "target",
        "validation_regime",
        "protocol",
        "model",
        "mean_r2",
        "mean_rmse",
        "mean_mae",
        "mean_median_ae",
        "pooled_r2",
        "pooled_rmse",
        "pooled_mae",
        "pooled_median_ae",
        "mean_r2_drop_vs_random",
        "mean_rmse_increase_vs_random",
        "mean_mae_increase_vs_random",
        "pooled_r2_drop_vs_random",
        "pooled_rmse_increase_vs_random",
        "pooled_mae_increase_vs_random",
    ]
]


# ------------------------------------------------------------
# 4) Robustness ranking across grouped validation regimes
# ------------------------------------------------------------
# This table is used only as a transparent selection aid.
# It is not a new performance metric and should not be overclaimed.

grouped_summary = summary[summary["is_grouped"]].copy()

grouped_robustness = (
    grouped_summary
    .groupby(["target", "protocol", "model"], as_index=False)
    .agg(
        n_grouped_regimes=("validation_regime", "nunique"),
        grouped_mean_r2=("mean_r2", "mean"),
        grouped_mean_rmse=("mean_rmse", "mean"),
        grouped_mean_mae=("mean_mae", "mean"),
        grouped_pooled_r2=("pooled_r2", "mean"),
        grouped_pooled_rmse=("pooled_rmse", "mean"),
        grouped_pooled_mae=("pooled_mae", "mean"),
        grouped_mean_r2_drop=("mean_r2_drop_vs_random", "mean"),
        grouped_rmse_increase=("mean_rmse_increase_vs_random", "mean"),
        grouped_mae_increase=("mean_mae_increase_vs_random", "mean"),
    )
)

grouped_robustness = grouped_robustness.sort_values(
    ["target", "grouped_mean_rmse"],
    ascending=[True, True]
)

grouped_robustness["rank_by_grouped_rmse"] = (
    grouped_robustness
    .groupby("target")["grouped_mean_rmse"]
    .rank(method="dense", ascending=True)
    .astype(int)
)


# ------------------------------------------------------------
# 5) Core grouped robustness ranking
# ------------------------------------------------------------
# Core grouped regimes exclude family-grouped validation because
# feedstock-family grouping has only a small number of groups and
# should be treated as supplementary.

core_grouped_summary = summary[summary["is_core_grouped"]].copy()

core_grouped_robustness = (
    core_grouped_summary
    .groupby(["target", "protocol", "model"], as_index=False)
    .agg(
        n_core_grouped_regimes=("validation_regime", "nunique"),
        core_mean_r2=("mean_r2", "mean"),
        core_mean_rmse=("mean_rmse", "mean"),
        core_mean_mae=("mean_mae", "mean"),
        core_pooled_r2=("pooled_r2", "mean"),
        core_pooled_rmse=("pooled_rmse", "mean"),
        core_pooled_mae=("pooled_mae", "mean"),
        core_mean_r2_drop=("mean_r2_drop_vs_random", "mean"),
        core_rmse_increase=("mean_rmse_increase_vs_random", "mean"),
        core_mae_increase=("mean_mae_increase_vs_random", "mean"),
    )
)

core_grouped_robustness = core_grouped_robustness.sort_values(
    ["target", "core_mean_rmse"],
    ascending=[True, True]
)

core_grouped_robustness["rank_by_core_grouped_rmse"] = (
    core_grouped_robustness
    .groupby("target")["core_mean_rmse"]
    .rank(method="dense", ascending=True)
    .astype(int)
)


# ------------------------------------------------------------
# 6) Random-CV champions
# ------------------------------------------------------------

random_champions = (
    summary[summary["validation_regime"] == RANDOM_REGIME]
    .sort_values(["target", "mean_rmse"])
    .groupby("target", as_index=False)
    .head(5)
    .copy()
)

random_champions = random_champions[
    [
        "target",
        "protocol",
        "model",
        "mean_r2",
        "mean_rmse",
        "mean_mae",
        "pooled_r2",
        "pooled_rmse",
        "pooled_mae",
    ]
]


# ------------------------------------------------------------
# 7) Official model-selection decision table
# ------------------------------------------------------------
# These decisions are based on Phase 7 results and are intentionally
# conservative. They should be reviewed after uncertainty and
# applicability-domain analyses.

official_decisions = pd.DataFrame([
    {
        "target": "biochar",
        "role": "primary_model_for_next_phases",
        "selected_model": "ExtraTrees",
        "selected_protocol": "A_numeric_only",
        "reason": (
            "Best or near-best performance across source-, feedstock-, and "
            "family-grouped validation; does not rely on exact feedstock identity; "
            "most defensible for unseen-feedstock settings."
        ),
        "claim_strength": "moderate",
        "remaining_risk": (
            "Performance still drops relative to random K-fold; uncertainty and "
            "applicability-domain checks are still required."
        ),
    },
    {
        "target": "biochar",
        "role": "random_cv_champion_reference",
        "selected_model": "HistGradientBoosting",
        "selected_protocol": "A_numeric_only",
        "reason": (
            "Best random K-fold performance; retained only as a reference showing "
            "that the random-CV champion is not necessarily the most robust grouped-CV model."
        ),
        "claim_strength": "comparison_only",
        "remaining_risk": (
            "Should not be selected as the final reliability model based only on random-CV accuracy."
        ),
    },
    {
        "target": "bio_oil",
        "role": "primary_model_for_next_phases",
        "selected_model": "ExtraTrees",
        "selected_protocol": "C_feedstock_family",
        "reason": (
            "Best grouped-average RMSE and best feedstock-/family-grouped performance. "
            "However, absolute grouped performance remains weak, so this is selected "
            "as the least-bad reliability-screening model, not as a high-confidence predictor."
        ),
        "claim_strength": "weak_to_moderate",
        "remaining_risk": (
            "Grouped-validation R2 remains low; bio-oil prediction likely suffers from "
            "unreported process variables such as vapor residence time, condenser design, "
            "and liquid collection protocol."
        ),
    },
    {
        "target": "bio_oil",
        "role": "source_grouped_sensitivity_model",
        "selected_model": "ExtraTrees",
        "selected_protocol": "A_numeric_only",
        "reason": (
            "Best source-grouped performance for bio-oil and nearly tied with family "
            "protocol under family-grouped validation. Kept as sensitivity model."
        ),
        "claim_strength": "sensitivity_only",
        "remaining_risk": (
            "Still weak under grouped validation; should not be used for strong optimization claims."
        ),
    },
    {
        "target": "bio_oil",
        "role": "random_cv_champion_reference",
        "selected_model": "HistGradientBoosting",
        "selected_protocol": "B_feedstock_identity",
        "reason": (
            "Best random K-fold performance; retained to demonstrate the optimism gap "
            "between random and grouped validation."
        ),
        "claim_strength": "comparison_only",
        "remaining_risk": (
            "Random-CV performance does not transfer to source-, feedstock-, or "
            "family-grouped validation."
        ),
    },
])


# ------------------------------------------------------------
# 8) Optional prediction-level diagnostics for selected models
# ------------------------------------------------------------

selected_error_by_group = None

if PREDICTION_PATH.exists():
    predictions = pd.read_excel(PREDICTION_PATH)

    required_pred_cols = [
        "target",
        "protocol",
        "model",
        "validation_regime",
        "y_true",
        "y_pred",
        "signed_error",
        "abs_error",
        "source_group",
        "source_pair_group",
        "feedstock",
        "feedstock_family",
    ]

    missing_pred = [col for col in required_pred_cols if col not in predictions.columns]

    if missing_pred:
        print("Prediction file exists, but required columns are missing:", missing_pred)
    else:
        selected_specs = [
            ("biochar", "A_numeric_only", "ExtraTrees"),
            ("bio_oil", "C_feedstock_family", "ExtraTrees"),
            ("bio_oil", "A_numeric_only", "ExtraTrees"),
        ]

        rows = []

        for target, protocol, model in selected_specs:
            subset = predictions[
                (predictions["target"] == target) &
                (predictions["protocol"] == protocol) &
                (predictions["model"] == model) &
                (predictions["validation_regime"].isin(GROUPED_REGIMES))
            ].copy()

            if subset.empty:
                continue

            for group_col in ["source_group", "source_pair_group", "feedstock", "feedstock_family"]:
                tmp = (
                    subset
                    .groupby(["target", "protocol", "model", "validation_regime", group_col], as_index=False)
                    .agg(
                        n=("abs_error", "size"),
                        mean_y_true=("y_true", "mean"),
                        mean_y_pred=("y_pred", "mean"),
                        mean_signed_error=("signed_error", "mean"),
                        mean_abs_error=("abs_error", "mean"),
                        median_abs_error=("abs_error", "median"),
                        max_abs_error=("abs_error", "max"),
                    )
                    .rename(columns={group_col: "group_value"})
                )

                tmp["group_type"] = group_col
                rows.append(tmp)

        if rows:
            selected_error_by_group = pd.concat(rows, ignore_index=True)
            selected_error_by_group = selected_error_by_group.sort_values(
                ["target", "protocol", "model", "validation_regime", "group_type", "mean_abs_error"],
                ascending=[True, True, True, True, True, False]
            )
        else:
            print("No selected prediction subsets were found.")
else:
    print("Prediction file not found. Skipping prediction-level diagnostics.")


# ------------------------------------------------------------
# 9) Save Phase 8 outputs
# ------------------------------------------------------------

best_by_regime.to_excel("phase8_best_by_regime.xlsx", index=False)
grouped_robustness.to_excel("phase8_grouped_robustness_ranking.xlsx", index=False)
core_grouped_robustness.to_excel("phase8_core_grouped_robustness_ranking.xlsx", index=False)
random_champions.to_excel("phase8_random_cv_champions.xlsx", index=False)
official_decisions.to_excel("phase8_official_model_decisions.xlsx", index=False)

if selected_error_by_group is not None:
    selected_error_by_group.to_excel("phase8_selected_error_by_group.xlsx", index=False)

print("\nSaved Phase 8 files:")
print("- phase8_best_by_regime.xlsx")
print("- phase8_grouped_robustness_ranking.xlsx")
print("- phase8_core_grouped_robustness_ranking.xlsx")
print("- phase8_random_cv_champions.xlsx")
print("- phase8_official_model_decisions.xlsx")

if selected_error_by_group is not None:
    print("- phase8_selected_error_by_group.xlsx")


# ------------------------------------------------------------
# 10) Compact console report
# ------------------------------------------------------------

print("\nOfficial model decisions:")
print(
    official_decisions[
        ["target", "role", "selected_model", "selected_protocol", "claim_strength"]
    ].to_string(index=False)
)

print("\nTop grouped robustness ranking:")
cols = [
    "target",
    "protocol",
    "model",
    "grouped_mean_r2",
    "grouped_mean_rmse",
    "grouped_mean_mae",
    "rank_by_grouped_rmse",
]

display_table = grouped_robustness[cols].copy()

for col in ["grouped_mean_r2", "grouped_mean_rmse", "grouped_mean_mae"]:
    display_table[col] = display_table[col].round(3)

print(display_table.head(12).to_string(index=False))
