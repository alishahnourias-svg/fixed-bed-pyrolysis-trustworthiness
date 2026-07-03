# ============================================================
# Phase 11-B2 - Reliability-Aware Candidate Scenario Ranking
# Project:
# From Accuracy to Trustworthiness:
# Grouped Validation, Applicability-Domain Mapping,
# and Reliability-Aware Optimization for ML-Based
# Fixed-Bed Biomass Pyrolysis Yield Prediction
# ============================================================
#
# Purpose:
# This script does NOT claim to find true optimum pyrolysis conditions.
# It generates candidate scenarios from empirically supported biochar groups
# and ranks them using:
#   1) predicted biochar yield,
#   2) conformal lower confidence bound,
#   3) applicability-domain class,
#   4) conservative reliability filters.
#
# Main target:
#   Biochar yield only.
#
# Bio-oil is intentionally excluded from this optimization script because
# previous phases showed weak grouped-validation performance and weak
# uncertainty-error association for bio-oil.
# ============================================================

import inspect
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error


# ------------------------------------------------------------
# 0) General settings
# ------------------------------------------------------------

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

N_CANDIDATES_PER_GROUP = 3000
CALIBRATION_SIZE = 0.20
NOMINAL_COVERAGE = 0.90
K_NEIGHBORS = 5

DATA_PATH = Path("clean_audit_data_with_family.xlsx")
SCOPE_PATH = Path("phase11_recommended_scope.xlsx")
SCENARIO_SPACE_PATH = Path("phase11_scenario_space_definitions.xlsx")

assert DATA_PATH.exists(), f"File not found: {DATA_PATH.resolve()}"
assert SCOPE_PATH.exists(), f"File not found: {SCOPE_PATH.resolve()}"
assert SCENARIO_SPACE_PATH.exists(), f"File not found: {SCENARIO_SPACE_PATH.resolve()}"

df = pd.read_excel(DATA_PATH)
scope = pd.read_excel(SCOPE_PATH)
scenario_space = pd.read_excel(SCENARIO_SPACE_PATH)

df = df.reset_index(drop=True)
df["row_id"] = df.index.astype(int)
df["excel_row_number"] = df.index.astype(int) + 2

print("Loaded data:", DATA_PATH, df.shape)
print("Loaded scope:", SCOPE_PATH, scope.shape)
print("Loaded scenario space:", SCENARIO_SPACE_PATH, scenario_space.shape)


# ------------------------------------------------------------
# 1) Column definitions
# ------------------------------------------------------------

TARGET_COL = "biochar_yield"

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

FEATURES = COMPOSITION_FEATURES + OPERATIONAL_FEATURES

CATEGORICAL_FEATURES = []

GROUP_COLUMNS = [
    "feedstock",
    "feedstock_family",
    "source_group",
    "source_pair_group",
]

required_cols = [TARGET_COL] + FEATURES + GROUP_COLUMNS
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns in data: {missing_cols}")

print("Required-column check passed.")


# ------------------------------------------------------------
# 2) Utility functions
# ------------------------------------------------------------

def safe_group_string(series):
    return (
        series
        .astype("string")
        .fillna("MISSING")
        .str.replace(r"\.0$", "", regex=True)
    )


for col in GROUP_COLUMNS:
    df[col] = safe_group_string(df[col])


def make_onehot_encoder():
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [
        ("numeric", numeric_transformer, FEATURES)
    ]

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model():
    return ExtraTreesRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=1,
        min_samples_leaf=2,
    )


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        return np.nan

    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n

    if q_level >= 1:
        return float(np.max(scores))

    try:
        return float(np.quantile(scores, q_level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, q_level, interpolation="higher"))


def predict_extra_trees_mean_std(pipe, X):
    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    X_transformed = preprocessor.transform(X)

    tree_preds = np.column_stack([
        tree.predict(X_transformed) for tree in model.estimators_
    ])

    mean_pred = np.mean(tree_preds, axis=1)
    std_pred = np.std(tree_preds, axis=1, ddof=1)

    return mean_pred, std_pred


def classify_by_train_threshold(score, q75, q95):
    if not np.isfinite(score):
        return "unknown"
    if score <= q75:
        return "in_domain"
    if score <= q95:
        return "near_domain"
    return "out_of_domain"


def compute_knn_scores(X_train, X_query, k=K_NEIGHBORS):
    n_train = X_train.shape[0]
    effective_k = min(k, max(1, n_train - 1))

    train_neighbors = min(effective_k + 1, n_train)
    nn_train = NearestNeighbors(n_neighbors=train_neighbors, metric="euclidean")
    nn_train.fit(X_train)

    train_distances, _ = nn_train.kneighbors(X_train)
    if train_distances.shape[1] > 1:
        train_scores = train_distances[:, 1:].mean(axis=1)
    else:
        train_scores = train_distances[:, 0]

    nn_query = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
    nn_query.fit(X_train)

    query_distances, _ = nn_query.kneighbors(X_query)
    query_scores = query_distances.mean(axis=1)

    return train_scores, query_scores


def compute_mahalanobis_scores(X_train, X_query):
    cov = LedoitWolf().fit(X_train)
    precision = cov.precision_
    center = cov.location_

    def distances(X):
        diff = X - center
        squared = np.einsum("ij,jk,ik->i", diff, precision, diff)
        squared = np.maximum(squared, 0)
        return np.sqrt(squared)

    return distances(X_train), distances(X_query)


def get_combined_ad_class(knn_class, maha_class):
    if "out_of_domain" in [knn_class, maha_class]:
        return "out_of_domain"
    if "near_domain" in [knn_class, maha_class]:
        return "near_domain"
    if knn_class == "in_domain" and maha_class == "in_domain":
        return "in_domain"
    return "unknown"


def empirical_quantile(series, q):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.quantile(q))


def choose_operational_value(base_value, lower, upper, rng):
    """
    Samples an operational variable around the empirical scenario range.

    If a range is valid, sample uniformly inside q10-q90.
    If the range is not valid, keep the base value.
    """
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
        return float(rng.uniform(lower, upper))

    if pd.notna(base_value):
        return float(base_value)

    return np.nan


# ------------------------------------------------------------
# 3) Prepare biochar training/calibration data
# ------------------------------------------------------------

biochar_df = df[df[TARGET_COL].notna()].copy()

X_all = biochar_df[FEATURES].copy()
y_all = biochar_df[TARGET_COL].astype(float).copy()
groups_all = biochar_df["source_group"].copy()

# Group-aware train/calibration split by source group.
# This keeps complete literature sources in either proper-training or calibration.
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=CALIBRATION_SIZE,
    random_state=RANDOM_STATE,
)

proper_idx, calib_idx = next(
    gss.split(X_all, y_all, groups=groups_all)
)

X_proper = X_all.iloc[proper_idx].copy()
y_proper = y_all.iloc[proper_idx].copy()

X_calib = X_all.iloc[calib_idx].copy()
y_calib = y_all.iloc[calib_idx].copy()

pipe = Pipeline(
    steps=[
        ("preprocess", make_preprocessor()),
        ("model", make_model()),
    ]
)

pipe.fit(X_proper, y_proper)

calib_pred, calib_std = predict_extra_trees_mean_std(pipe, X_calib)
calib_abs_resid = np.abs(y_calib.values - calib_pred)

alpha = 1 - NOMINAL_COVERAGE
q_abs_90 = conformal_quantile(calib_abs_resid, alpha=alpha)

calib_metrics = calculate_metrics(y_calib, calib_pred)

print("\nCalibration performance on held-out source groups:")
print(calib_metrics)
print(f"Absolute conformal q90 half-width: {q_abs_90:.3f}")


# ------------------------------------------------------------
# 4) Fit applicability-domain objects on proper-training data
# ------------------------------------------------------------

preprocessor = pipe.named_steps["preprocess"]
X_proper_transformed = np.asarray(preprocessor.transform(X_proper), dtype=float)

train_knn_scores, _ = compute_knn_scores(X_proper_transformed, X_proper_transformed, k=K_NEIGHBORS)
train_maha_scores, _ = compute_mahalanobis_scores(X_proper_transformed, X_proper_transformed)

knn_q75, knn_q95 = np.quantile(train_knn_scores, [0.75, 0.95])
maha_q75, maha_q95 = np.quantile(train_maha_scores, [0.75, 0.95])

print("\nApplicability-domain thresholds:")
print("KNN q75/q95:", knn_q75, knn_q95)
print("Mahalanobis q75/q95:", maha_q75, maha_q95)


# ------------------------------------------------------------
# 5) Select eligible biochar groups
# ------------------------------------------------------------

eligible_biochar_scope = scope[
    (scope["target"] == "biochar")
    & (scope["recommended_role"] == "main_reliability_aware_scenario_ranking")
].copy()

# Conservative main analysis:
# use feedstock families as the main scenario-ranking level.
main_family_scope = eligible_biochar_scope[
    eligible_biochar_scope["group_type"] == "feedstock_family"
].copy()

# Optional strong feedstock case studies:
# use only feedstocks with at least 50 rows and at least 4 sources.
case_feedstock_scope = eligible_biochar_scope[
    (eligible_biochar_scope["group_type"] == "feedstock")
    & (eligible_biochar_scope["n_rows"] >= 50)
    & (eligible_biochar_scope["n_sources"] >= 4)
].copy()

print("\nMain eligible biochar families:", len(main_family_scope))
print("Optional stronger biochar feedstock cases:", len(case_feedstock_scope))


# ------------------------------------------------------------
# 6) Candidate scenario generation
# ------------------------------------------------------------

def generate_candidates_for_group(group_type, group_value, n_candidates=N_CANDIDATES_PER_GROUP):
    """
    Generates candidate scenarios for one eligible biochar group.

    Conservative design:
    - Sample a real observed row as the base scenario.
    - Keep composition variables from the real base row.
    - Sample only operational variables inside empirical q10-q90 ranges
      of the same group.
    """
    subset = biochar_df[biochar_df[group_type] == group_value].copy()

    if subset.empty:
        return pd.DataFrame()

    scenario_def = scenario_space[
        (scenario_space["target"] == "biochar")
        & (scenario_space["group_type"] == group_type)
        & (scenario_space["group_value"] == group_value)
    ].copy()

    if scenario_def.empty:
        return pd.DataFrame()

    scenario_def = scenario_def.iloc[0].to_dict()

    # Use only rows with at least some composition and temperature information.
    base_pool = subset.copy()

    if base_pool.empty:
        return pd.DataFrame()

    sampled_indices = rng.choice(
        base_pool.index.to_numpy(),
        size=n_candidates,
        replace=True,
    )

    rows = []

    for candidate_id, base_index in enumerate(sampled_indices, start=1):
        base = base_pool.loc[base_index].copy()

        row = {
            "candidate_id": candidate_id,
            "group_type": group_type,
            "group_value": group_value,
            "base_row_id": int(base["row_id"]),
            "base_excel_row_number": int(base["excel_row_number"]),
            "base_feedstock": base["feedstock"],
            "base_feedstock_family": base["feedstock_family"],
            "base_source_group": base["source_group"],
            "base_source_pair_group": base["source_pair_group"],
            "observed_target_yield": base[TARGET_COL],
        }

        # Keep composition fixed from real observed base row.
        for col in COMPOSITION_FEATURES:
            row[col] = base[col]

        # Sample operational variables within empirical q10-q90 group range.
        for col in OPERATIONAL_FEATURES:
            lower = scenario_def.get(f"{col}_lower_q10", np.nan)
            upper = scenario_def.get(f"{col}_upper_q90", np.nan)
            row[col] = choose_operational_value(base[col], lower, upper, rng)

            row[f"{col}_group_q10"] = lower
            row[f"{col}_group_q90"] = upper

        rows.append(row)

    return pd.DataFrame(rows)


candidate_tables = []

for _, row in main_family_scope.iterrows():
    candidate_tables.append(
        generate_candidates_for_group(
            group_type=row["group_type"],
            group_value=row["group_value"],
            n_candidates=N_CANDIDATES_PER_GROUP,
        )
    )

for _, row in case_feedstock_scope.iterrows():
    candidate_tables.append(
        generate_candidates_for_group(
            group_type=row["group_type"],
            group_value=row["group_value"],
            n_candidates=N_CANDIDATES_PER_GROUP,
        )
    )

phase11_candidate_scenarios = pd.concat(candidate_tables, ignore_index=True)

print("\nGenerated candidate scenarios:", phase11_candidate_scenarios.shape)


# ------------------------------------------------------------
# 7) Predict candidates, compute uncertainty and AD class
# ------------------------------------------------------------

X_candidates = phase11_candidate_scenarios[FEATURES].copy()

candidate_pred, candidate_std = predict_extra_trees_mean_std(pipe, X_candidates)

phase11_candidate_scenarios["predicted_biochar_yield"] = candidate_pred
phase11_candidate_scenarios["ensemble_std"] = candidate_std

# Absolute conformal interval with constant half-width from held-out calibration sources.
phase11_candidate_scenarios["conformal_half_width_90"] = q_abs_90
phase11_candidate_scenarios["lower_bound_90"] = (
    phase11_candidate_scenarios["predicted_biochar_yield"]
    - phase11_candidate_scenarios["conformal_half_width_90"]
)
phase11_candidate_scenarios["upper_bound_90"] = (
    phase11_candidate_scenarios["predicted_biochar_yield"]
    + phase11_candidate_scenarios["conformal_half_width_90"]
)

# AD scoring for candidates
X_candidates_transformed = np.asarray(preprocessor.transform(X_candidates), dtype=float)

_, candidate_knn = compute_knn_scores(X_proper_transformed, X_candidates_transformed, k=K_NEIGHBORS)
_, candidate_maha = compute_mahalanobis_scores(X_proper_transformed, X_candidates_transformed)

phase11_candidate_scenarios["knn_mean_distance"] = candidate_knn
phase11_candidate_scenarios["mahalanobis_distance"] = candidate_maha

phase11_candidate_scenarios["knn_ad_class"] = [
    classify_by_train_threshold(x, knn_q75, knn_q95)
    for x in candidate_knn
]

phase11_candidate_scenarios["mahalanobis_ad_class"] = [
    classify_by_train_threshold(x, maha_q75, maha_q95)
    for x in candidate_maha
]

phase11_candidate_scenarios["combined_ad_class"] = [
    get_combined_ad_class(kc, mc)
    for kc, mc in zip(
        phase11_candidate_scenarios["knn_ad_class"],
        phase11_candidate_scenarios["mahalanobis_ad_class"],
    )
]

phase11_candidate_scenarios["ad_acceptable"] = (
    phase11_candidate_scenarios["combined_ad_class"].isin(["in_domain", "near_domain"])
)

# Reliability score:
# Conservative lower bound used as the main score.
# Out-of-domain scenarios are not deleted from the full candidate table,
# but they are excluded from the main recommended list.
phase11_candidate_scenarios["reliable_score_lcb90"] = phase11_candidate_scenarios["lower_bound_90"]


# ------------------------------------------------------------
# 8) Rank scenarios
# ------------------------------------------------------------

# Yield-only ranking: intentionally retained as a comparison and warning.
yield_only_top = (
    phase11_candidate_scenarios
    .sort_values(["group_type", "group_value", "predicted_biochar_yield"], ascending=[True, True, False])
    .groupby(["group_type", "group_value"], as_index=False)
    .head(25)
    .copy()
)

# Reliability-aware ranking: AD acceptable + top lower-bound score.
reliable_pool = phase11_candidate_scenarios[
    phase11_candidate_scenarios["ad_acceptable"]
].copy()

reliability_top = (
    reliable_pool
    .sort_values(["group_type", "group_value", "reliable_score_lcb90"], ascending=[True, True, False])
    .groupby(["group_type", "group_value"], as_index=False)
    .head(25)
    .copy()
)

yield_only_top["ranking_type"] = "yield_only"
reliability_top["ranking_type"] = "reliability_aware_lcb90_ad_filtered"

phase11_top_ranked_scenarios = pd.concat(
    [yield_only_top, reliability_top],
    ignore_index=True,
)


# ------------------------------------------------------------
# 9) Build operational windows from reliability-aware top scenarios
# ------------------------------------------------------------

window_rows = []

if len(reliability_top) > 0:
    for keys, group in reliability_top.groupby(["group_type", "group_value"]):
        group_type, group_value = keys

        row = {
            "group_type": group_type,
            "group_value": group_value,
            "n_top_scenarios": len(group),
            "mean_predicted_biochar_yield": group["predicted_biochar_yield"].mean(),
            "median_predicted_biochar_yield": group["predicted_biochar_yield"].median(),
            "mean_lower_bound_90": group["lower_bound_90"].mean(),
            "median_lower_bound_90": group["lower_bound_90"].median(),
            "mean_ensemble_std": group["ensemble_std"].mean(),
            "fraction_in_domain": (group["combined_ad_class"] == "in_domain").mean(),
            "fraction_near_domain": (group["combined_ad_class"] == "near_domain").mean(),
            "fraction_out_of_domain": (group["combined_ad_class"] == "out_of_domain").mean(),
        }

        for col in OPERATIONAL_FEATURES:
            row[f"{col}_p10"] = empirical_quantile(group[col], 0.10)
            row[f"{col}_median"] = empirical_quantile(group[col], 0.50)
            row[f"{col}_p90"] = empirical_quantile(group[col], 0.90)

        window_rows.append(row)

phase11_reliable_operating_windows = pd.DataFrame(window_rows)


# ------------------------------------------------------------
# 10) Compare yield-only vs reliability-aware rankings
# ------------------------------------------------------------

comparison_rows = []

for keys, group in phase11_top_ranked_scenarios.groupby(["group_type", "group_value", "ranking_type"]):
    group_type, group_value, ranking_type = keys

    row = {
        "group_type": group_type,
        "group_value": group_value,
        "ranking_type": ranking_type,
        "n_scenarios": len(group),
        "mean_predicted_biochar_yield": group["predicted_biochar_yield"].mean(),
        "median_predicted_biochar_yield": group["predicted_biochar_yield"].median(),
        "mean_lower_bound_90": group["lower_bound_90"].mean(),
        "median_lower_bound_90": group["lower_bound_90"].median(),
        "mean_ensemble_std": group["ensemble_std"].mean(),
        "fraction_in_domain": (group["combined_ad_class"] == "in_domain").mean(),
        "fraction_near_domain": (group["combined_ad_class"] == "near_domain").mean(),
        "fraction_out_of_domain": (group["combined_ad_class"] == "out_of_domain").mean(),
        "mean_knn_distance": group["knn_mean_distance"].mean(),
        "mean_mahalanobis_distance": group["mahalanobis_distance"].mean(),
    }

    comparison_rows.append(row)

phase11_yield_vs_reliable_comparison = pd.DataFrame(comparison_rows)


# ------------------------------------------------------------
# 11) Candidate AD distribution by group
# ------------------------------------------------------------

phase11_candidate_ad_distribution = (
    phase11_candidate_scenarios
    .groupby(["group_type", "group_value", "combined_ad_class"], as_index=False)
    .agg(n=("candidate_id", "size"))
)

totals = (
    phase11_candidate_ad_distribution
    .groupby(["group_type", "group_value"], as_index=False)
    ["n"].sum()
    .rename(columns={"n": "total_n"})
)

phase11_candidate_ad_distribution = phase11_candidate_ad_distribution.merge(
    totals,
    on=["group_type", "group_value"],
    how="left",
)

phase11_candidate_ad_distribution["fraction"] = (
    phase11_candidate_ad_distribution["n"] / phase11_candidate_ad_distribution["total_n"]
)


# ------------------------------------------------------------
# 12) Save outputs
# ------------------------------------------------------------

phase11_candidate_scenarios.to_excel("phase11b2_candidate_scenarios.xlsx", index=False)
phase11_top_ranked_scenarios.to_excel("phase11b2_top_ranked_scenarios.xlsx", index=False)
phase11_reliable_operating_windows.to_excel("phase11b2_reliable_operating_windows.xlsx", index=False)
phase11_yield_vs_reliable_comparison.to_excel("phase11b2_yield_vs_reliable_comparison.xlsx", index=False)
phase11_candidate_ad_distribution.to_excel("phase11b2_candidate_ad_distribution.xlsx", index=False)

calibration_record = pd.DataFrame([{
    "target": "biochar",
    "model": "ExtraTrees",
    "protocol": "A_numeric_only",
    "calibration_split": "GroupShuffleSplit_by_source_group",
    "n_proper_train": len(X_proper),
    "n_calibration": len(X_calib),
    "calib_r2": calib_metrics["r2"],
    "calib_rmse": calib_metrics["rmse"],
    "calib_mae": calib_metrics["mae"],
    "calib_median_ae": calib_metrics["median_ae"],
    "nominal_coverage": NOMINAL_COVERAGE,
    "absolute_conformal_half_width": q_abs_90,
    "knn_q75": knn_q75,
    "knn_q95": knn_q95,
    "mahalanobis_q75": maha_q75,
    "mahalanobis_q95": maha_q95,
}])

calibration_record.to_excel("phase11b2_model_calibration_record.xlsx", index=False)

print("\nSaved Phase 11-B2 files:")
print("- phase11b2_candidate_scenarios.xlsx")
print("- phase11b2_top_ranked_scenarios.xlsx")
print("- phase11b2_reliable_operating_windows.xlsx")
print("- phase11b2_yield_vs_reliable_comparison.xlsx")
print("- phase11b2_candidate_ad_distribution.xlsx")
print("- phase11b2_model_calibration_record.xlsx")


# ------------------------------------------------------------
# 13) Console preview
# ------------------------------------------------------------

print("\nCandidate AD distribution:")
preview_ad = phase11_candidate_ad_distribution.copy()
preview_ad["fraction"] = preview_ad["fraction"].round(3)
print(preview_ad.to_string(index=False))

print("\nYield-only vs reliability-aware comparison:")
preview_comp = phase11_yield_vs_reliable_comparison.copy()
for col in [
    "mean_predicted_biochar_yield",
    "mean_lower_bound_90",
    "mean_ensemble_std",
    "fraction_in_domain",
    "fraction_near_domain",
    "fraction_out_of_domain",
]:
    if col in preview_comp.columns:
        preview_comp[col] = preview_comp[col].round(3)

print(
    preview_comp.sort_values(["group_type", "group_value", "ranking_type"])
    .to_string(index=False)
)

print("\nReliable operating windows preview:")
preview_windows = phase11_reliable_operating_windows.copy()
for col in preview_windows.select_dtypes(include=[np.number]).columns:
    preview_windows[col] = preview_windows[col].round(3)

print(
    preview_windows
    .sort_values(["group_type", "group_value"])
    .to_string(index=False)
)

print("\nImportant interpretation note:")
print("These are reliability-aware candidate rankings, not experimentally validated optimum conditions.")
print("Report operating windows, not exact optimum points.")
