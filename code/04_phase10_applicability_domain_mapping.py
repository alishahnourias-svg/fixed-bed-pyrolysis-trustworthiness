# ============================================================
# Phase 10 - Applicability-Domain Mapping and Reliability Filtering
# Project:
# From Accuracy to Trustworthiness:
# Grouped Validation, Applicability-Domain Mapping,
# and Reliability-Aware Optimization for ML-Based
# Fixed-Bed Biomass Pyrolysis Yield Prediction
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
from sklearn.model_selection import KFold, GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr, pearsonr


# ------------------------------------------------------------
# 0) General settings
# ------------------------------------------------------------

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
K_NEIGHBORS = 5

DATA_PATH = Path("clean_audit_data_with_family.xlsx")
PREDICTION_PATH = Path("phase7_v2_prediction_results.xlsx")
PHASE9_INTERVAL_PATH = Path("phase9_prediction_intervals.xlsx")

assert DATA_PATH.exists(), f"File not found: {DATA_PATH.resolve()}"
assert PREDICTION_PATH.exists(), f"File not found: {PREDICTION_PATH.resolve()}"

df = pd.read_excel(DATA_PATH)
df = df.reset_index(drop=True)
df["row_id"] = df.index.astype(int)
df["excel_row_number"] = df.index.astype(int) + 2

predictions = pd.read_excel(PREDICTION_PATH)

print("Loaded:", DATA_PATH)
print("Data shape:", df.shape)
print("Loaded:", PREDICTION_PATH)
print("Prediction shape:", predictions.shape)


# ------------------------------------------------------------
# 1) Feature protocols and selected Phase-8 model decisions
# ------------------------------------------------------------

NUMERIC_FEATURES = [
    "carbon",
    "hydrogen",
    "nitrogen",
    "oxygen",
    "moisture",
    "volatile_matter",
    "fixed_carbon",
    "ash",
    "particle_size_mm",
    "temperature_c",
    "heating_rate_c_min",
    "gas_flow_rate_l_min",
    "residence_time_min",
    "raw_material_supply_g",
]

TARGETS = {
    "biochar": "biochar_yield",
    "bio_oil": "bio_oil_yield",
}

PROTOCOLS = {
    "A_numeric_only": {
        "numeric": NUMERIC_FEATURES,
        "categorical": [],
    },
    "C_feedstock_family": {
        "numeric": NUMERIC_FEATURES,
        "categorical": ["feedstock_family"],
    },
}

SELECTED_MODEL_SPECS = [
    {
        "target": "biochar",
        "protocol": "A_numeric_only",
        "model": "ExtraTrees",
        "role": "primary_biochar_model",
        "claim_strength": "moderate",
    },
    {
        "target": "bio_oil",
        "protocol": "C_feedstock_family",
        "model": "ExtraTrees",
        "role": "primary_bio_oil_screening_model",
        "claim_strength": "limited_screening",
    },
    {
        "target": "bio_oil",
        "protocol": "A_numeric_only",
        "model": "ExtraTrees",
        "role": "bio_oil_sensitivity_model",
        "claim_strength": "sensitivity_only",
    },
]

VALIDATION_REGIMES = [
    "random_kfold",
    "source_group_kfold",
    "feedstock_group_kfold",
    "family_group_kfold",
]


# ------------------------------------------------------------
# 2) Required-column checks
# ------------------------------------------------------------

required_data_cols = [
    "row_id",
    "excel_row_number",
    "feedstock",
    "feedstock_family",
    "source_group",
    "source_pair_group",
    "biochar_yield",
    "bio_oil_yield",
] + NUMERIC_FEATURES

missing_data_cols = [col for col in required_data_cols if col not in df.columns]
if missing_data_cols:
    raise ValueError(f"Missing required data columns: {missing_data_cols}")

required_pred_cols = [
    "target",
    "protocol",
    "model",
    "validation_regime",
    "fold",
    "row_id",
    "y_true",
    "y_pred",
    "signed_error",
    "abs_error",
    "source_group",
    "source_pair_group",
    "feedstock",
    "feedstock_family",
]

missing_pred_cols = [col for col in required_pred_cols if col not in predictions.columns]
if missing_pred_cols:
    raise ValueError(f"Missing required prediction columns: {missing_pred_cols}")

print("Required-column checks passed.")


# ------------------------------------------------------------
# 3) Helper functions
# ------------------------------------------------------------

def make_onehot_encoder():
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(protocol_name):
    if protocol_name not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol_name}")

    protocol = PROTOCOLS[protocol_name]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [
        ("numeric", numeric_transformer, protocol["numeric"])
    ]

    if protocol["categorical"]:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_transformer, protocol["categorical"]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def safe_group_string(series):
    return (
        series
        .astype("string")
        .fillna("MISSING")
        .str.replace(r"\.0$", "", regex=True)
    )


def get_model_data(dataframe, target_key, protocol_name):
    target_col = TARGETS[target_key]
    protocol = PROTOCOLS[protocol_name]
    feature_cols = protocol["numeric"] + protocol["categorical"]

    rows = dataframe[target_col].notna()

    X = dataframe.loc[rows, feature_cols].copy()
    y = dataframe.loc[rows, target_col].astype(float).copy()

    meta = pd.DataFrame({
        "row_id": dataframe.loc[rows, "row_id"].astype(int).values,
        "excel_row_number": dataframe.loc[rows, "excel_row_number"].astype(int).values,
        "source_group": safe_group_string(dataframe.loc[rows, "source_group"]).values,
        "source_pair_group": safe_group_string(dataframe.loc[rows, "source_pair_group"]).values,
        "feedstock": safe_group_string(dataframe.loc[rows, "feedstock"]).values,
        "feedstock_family": safe_group_string(dataframe.loc[rows, "feedstock_family"]).values,
    }, index=X.index)

    return X, y, meta


def get_cv_splits(X, y, meta, regime_name):
    if regime_name == "random_kfold":
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        return list(splitter.split(X, y))

    group_map = {
        "source_group_kfold": "source_group",
        "feedstock_group_kfold": "feedstock",
        "family_group_kfold": "feedstock_family",
    }

    if regime_name not in group_map:
        raise ValueError(f"Unknown validation regime: {regime_name}")

    group_col = group_map[regime_name]
    groups = meta[group_col]

    n_unique_groups = groups.nunique()
    effective_splits = min(N_SPLITS, n_unique_groups)

    if effective_splits < 2:
        raise ValueError(f"Not enough groups for {regime_name}: {n_unique_groups}")

    splitter = GroupKFold(n_splits=effective_splits)
    return list(splitter.split(X, y, groups=groups))


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def safe_corr(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan

    if method == "spearman":
        return float(spearmanr(x, y).correlation)
    if method == "pearson":
        return float(pearsonr(x, y)[0])

    raise ValueError(f"Unknown correlation method: {method}")


def classify_by_train_threshold(score, q75, q95):
    if not np.isfinite(score):
        return "unknown"
    if score <= q75:
        return "in_domain"
    if score <= q95:
        return "near_domain"
    return "out_of_domain"


def compute_knn_scores(X_train, X_test, k=K_NEIGHBORS):
    n_train = X_train.shape[0]
    effective_k = min(k, max(1, n_train - 1))

    # Training scores: use k+1 because the closest point is the point itself.
    train_neighbors = min(effective_k + 1, n_train)
    nn_train = NearestNeighbors(n_neighbors=train_neighbors, metric="euclidean")
    nn_train.fit(X_train)

    train_distances, _ = nn_train.kneighbors(X_train)

    if train_distances.shape[1] > 1:
        train_scores = train_distances[:, 1:].mean(axis=1)
    else:
        train_scores = train_distances[:, 0]

    # Test scores: mean distance to k nearest training observations.
    nn_test = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
    nn_test.fit(X_train)

    test_distances, _ = nn_test.kneighbors(X_test)
    test_scores = test_distances.mean(axis=1)

    return train_scores, test_scores


def compute_mahalanobis_scores(X_train, X_test):
    # Ledoit-Wolf shrinkage covariance is more stable than the empirical inverse covariance.
    cov = LedoitWolf().fit(X_train)
    precision = cov.precision_
    center = cov.location_

    def distances(X):
        diff = X - center
        squared = np.einsum("ij,jk,ik->i", diff, precision, diff)
        squared = np.maximum(squared, 0)
        return np.sqrt(squared)

    return distances(X_train), distances(X_test)


# ------------------------------------------------------------
# 4) Applicability-domain scoring for one selected spec and regime
# ------------------------------------------------------------

def compute_applicability_for_spec(target_key, protocol_name, model_name, role, claim_strength, regime_name):
    X, y, meta = get_model_data(df, target_key, protocol_name)
    splits = get_cv_splits(X, y, meta, regime_name)

    rows = []

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train_raw = X.iloc[train_idx].copy()
        X_test_raw = X.iloc[test_idx].copy()

        meta_train = meta.iloc[train_idx].copy()
        meta_test = meta.iloc[test_idx].copy()

        preprocessor = make_preprocessor(protocol_name)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        X_train = np.asarray(X_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)

        train_knn, test_knn = compute_knn_scores(X_train, X_test, k=K_NEIGHBORS)
        train_maha, test_maha = compute_mahalanobis_scores(X_train, X_test)

        knn_q75, knn_q95 = np.quantile(train_knn, [0.75, 0.95])
        maha_q75, maha_q95 = np.quantile(train_maha, [0.75, 0.95])

        # Combined class: a point is only in-domain if both scores are in-domain.
        # It is out-of-domain if either score is out-of-domain.
        knn_class = [
            classify_by_train_threshold(score, knn_q75, knn_q95)
            for score in test_knn
        ]

        maha_class = [
            classify_by_train_threshold(score, maha_q75, maha_q95)
            for score in test_maha
        ]

        combined_class = []
        for kc, mc in zip(knn_class, maha_class):
            if "out_of_domain" in [kc, mc]:
                combined_class.append("out_of_domain")
            elif "near_domain" in [kc, mc]:
                combined_class.append("near_domain")
            elif kc == "in_domain" and mc == "in_domain":
                combined_class.append("in_domain")
            else:
                combined_class.append("unknown")

        fold_df = pd.DataFrame({
            "target": target_key,
            "protocol": protocol_name,
            "model": model_name,
            "role": role,
            "claim_strength": claim_strength,
            "validation_regime": regime_name,
            "fold": fold_id,

            "row_id": meta_test["row_id"].values,
            "excel_row_number": meta_test["excel_row_number"].values,

            "knn_mean_distance": test_knn,
            "knn_train_q75": knn_q75,
            "knn_train_q95": knn_q95,
            "knn_ad_class": knn_class,

            "mahalanobis_distance": test_maha,
            "mahalanobis_train_q75": maha_q75,
            "mahalanobis_train_q95": maha_q95,
            "mahalanobis_ad_class": maha_class,

            "combined_ad_class": combined_class,

            "source_group": meta_test["source_group"].values,
            "source_pair_group": meta_test["source_pair_group"].values,
            "feedstock": meta_test["feedstock"].values,
            "feedstock_family": meta_test["feedstock_family"].values,
        })

        rows.append(fold_df)

    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------
# 5) Run Phase 10 AD scoring
# ------------------------------------------------------------

all_ad_rows = []
total_runs = len(SELECTED_MODEL_SPECS) * len(VALIDATION_REGIMES)
counter = 0

for spec in SELECTED_MODEL_SPECS:
    for regime in VALIDATION_REGIMES:
        counter += 1
        print(
            f"[{counter}/{total_runs}]",
            spec["target"],
            spec["protocol"],
            spec["role"],
            regime,
        )

        ad_df = compute_applicability_for_spec(
            target_key=spec["target"],
            protocol_name=spec["protocol"],
            model_name=spec["model"],
            role=spec["role"],
            claim_strength=spec["claim_strength"],
            regime_name=regime,
        )

        all_ad_rows.append(ad_df)

phase10_ad_scores = pd.concat(all_ad_rows, ignore_index=True)


# ------------------------------------------------------------
# 6) Merge AD scores with Phase-7 prediction errors
# ------------------------------------------------------------

selected_predictions = []

for spec in SELECTED_MODEL_SPECS:
    tmp = predictions[
        (predictions["target"] == spec["target"]) &
        (predictions["protocol"] == spec["protocol"]) &
        (predictions["model"] == spec["model"])
    ].copy()

    selected_predictions.append(tmp)

selected_predictions = pd.concat(selected_predictions, ignore_index=True)

merge_cols = [
    "target",
    "protocol",
    "model",
    "validation_regime",
    "fold",
    "row_id",
]

phase10_predictions = selected_predictions.merge(
    phase10_ad_scores,
    on=merge_cols,
    how="left",
    suffixes=("", "_ad"),
)

# Keep clean metadata columns when duplicates exist.
for col in ["source_group", "source_pair_group", "feedstock", "feedstock_family"]:
    ad_col = f"{col}_ad"
    if ad_col in phase10_predictions.columns:
        phase10_predictions[col] = phase10_predictions[col].fillna(phase10_predictions[ad_col])
        phase10_predictions = phase10_predictions.drop(columns=[ad_col])

# Add a reliability flag for simple filtering.
phase10_predictions["is_ad_acceptable"] = phase10_predictions["combined_ad_class"].isin(
    ["in_domain", "near_domain"]
)

phase10_predictions["is_strict_in_domain"] = phase10_predictions["combined_ad_class"].eq(
    "in_domain"
)


# ------------------------------------------------------------
# 7) Optional merge with Phase-9 uncertainty intervals at nominal 90%
# ------------------------------------------------------------

if PHASE9_INTERVAL_PATH.exists():
    phase9 = pd.read_excel(PHASE9_INTERVAL_PATH)

    needed_phase9 = [
        "target", "protocol", "model", "validation_regime", "fold", "row_id",
        "nominal_coverage", "ensemble_std",
        "covered_abs_conformal", "width_abs_conformal",
        "covered_norm_conformal", "width_norm_conformal",
    ]

    missing_phase9 = [col for col in needed_phase9 if col not in phase9.columns]

    if missing_phase9:
        print("Phase-9 file exists but missing columns:", missing_phase9)
    else:
        phase9_90 = phase9[np.isclose(phase9["nominal_coverage"], 0.90)].copy()

        phase10_predictions = phase10_predictions.merge(
            phase9_90[needed_phase9],
            on=merge_cols,
            how="left",
        )

        print("Merged Phase-9 90% interval diagnostics.")
else:
    print("Phase-9 prediction interval file not found. Skipping uncertainty merge.")


# ------------------------------------------------------------
# 8) Summary: AD class distribution and error by AD class
# ------------------------------------------------------------

summary_group_cols = [
    "target",
    "protocol",
    "model",
    "role",
    "claim_strength",
    "validation_regime",
    "combined_ad_class",
]

phase10_error_by_ad_class = (
    phase10_predictions
    .groupby(summary_group_cols, as_index=False)
    .agg(
        n=("abs_error", "size"),
        mean_y_true=("y_true", "mean"),
        mean_y_pred=("y_pred", "mean"),
        mean_signed_error=("signed_error", "mean"),
        mean_abs_error=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        rmse=("signed_error", rmse),
        max_abs_error=("abs_error", "max"),
        mean_knn_distance=("knn_mean_distance", "mean"),
        mean_mahalanobis_distance=("mahalanobis_distance", "mean"),
    )
)

# Add percentage within each target/protocol/model/regime.
totals = (
    phase10_error_by_ad_class
    .groupby(["target", "protocol", "model", "validation_regime"], as_index=False)
    ["n"].sum()
    .rename(columns={"n": "total_n"})
)

phase10_error_by_ad_class = phase10_error_by_ad_class.merge(
    totals,
    on=["target", "protocol", "model", "validation_regime"],
    how="left",
)

phase10_error_by_ad_class["class_fraction"] = (
    phase10_error_by_ad_class["n"] / phase10_error_by_ad_class["total_n"]
)


# ------------------------------------------------------------
# 9) Reliability filtering summary
# ------------------------------------------------------------

filter_rows = []

filter_definitions = {
    "all_predictions": lambda d: np.ones(len(d), dtype=bool),
    "in_or_near_domain_only": lambda d: d["combined_ad_class"].isin(["in_domain", "near_domain"]),
    "strict_in_domain_only": lambda d: d["combined_ad_class"].eq("in_domain"),
}

filter_group_cols = [
    "target",
    "protocol",
    "model",
    "role",
    "claim_strength",
    "validation_regime",
]

for keys, group in phase10_predictions.groupby(filter_group_cols):
    base = dict(zip(filter_group_cols, keys))
    total_n = len(group)

    for filter_name, filter_func in filter_definitions.items():
        mask = filter_func(group)
        subset = group[mask].copy()

        row = base.copy()
        row["filter_name"] = filter_name
        row["n_kept"] = len(subset)
        row["fraction_kept"] = len(subset) / total_n if total_n else np.nan

        if len(subset) == 0:
            row.update({
                "mean_abs_error": np.nan,
                "median_abs_error": np.nan,
                "rmse": np.nan,
                "mean_signed_error": np.nan,
                "mean_y_true": np.nan,
                "mean_y_pred": np.nan,
            })
        else:
            row.update({
                "mean_abs_error": subset["abs_error"].mean(),
                "median_abs_error": subset["abs_error"].median(),
                "rmse": rmse(subset["signed_error"].values),
                "mean_signed_error": subset["signed_error"].mean(),
                "mean_y_true": subset["y_true"].mean(),
                "mean_y_pred": subset["y_pred"].mean(),
            })

        # If uncertainty columns exist, summarize coverage and width after AD filtering.
        if "covered_abs_conformal" in subset.columns and len(subset) > 0:
            row["coverage_abs_90"] = subset["covered_abs_conformal"].mean()
            row["mean_width_abs_90"] = subset["width_abs_conformal"].mean()
            row["coverage_norm_90"] = subset["covered_norm_conformal"].mean()
            row["mean_width_norm_90"] = subset["width_norm_conformal"].mean()
            row["mean_ensemble_std"] = subset["ensemble_std"].mean()
        else:
            row["coverage_abs_90"] = np.nan
            row["mean_width_abs_90"] = np.nan
            row["coverage_norm_90"] = np.nan
            row["mean_width_norm_90"] = np.nan
            row["mean_ensemble_std"] = np.nan

        filter_rows.append(row)

phase10_reliability_filtering_summary = pd.DataFrame(filter_rows)


# ------------------------------------------------------------
# 10) AD score-error association
# ------------------------------------------------------------

association_rows = []

assoc_group_cols = [
    "target",
    "protocol",
    "model",
    "role",
    "validation_regime",
]

for keys, group in phase10_predictions.groupby(assoc_group_cols):
    row = dict(zip(assoc_group_cols, keys))

    row.update({
        "n": len(group),
        "spearman_knn_abs_error": safe_corr(group["knn_mean_distance"], group["abs_error"], method="spearman"),
        "pearson_knn_abs_error": safe_corr(group["knn_mean_distance"], group["abs_error"], method="pearson"),
        "spearman_mahalanobis_abs_error": safe_corr(group["mahalanobis_distance"], group["abs_error"], method="spearman"),
        "pearson_mahalanobis_abs_error": safe_corr(group["mahalanobis_distance"], group["abs_error"], method="pearson"),
        "mean_abs_error": group["abs_error"].mean(),
        "mean_knn_distance": group["knn_mean_distance"].mean(),
        "mean_mahalanobis_distance": group["mahalanobis_distance"].mean(),
    })

    if "ensemble_std" in group.columns:
        row["spearman_ensemble_std_abs_error"] = safe_corr(group["ensemble_std"], group["abs_error"], method="spearman")
    else:
        row["spearman_ensemble_std_abs_error"] = np.nan

    association_rows.append(row)

phase10_ad_error_association = pd.DataFrame(association_rows)


# ------------------------------------------------------------
# 11) Top out-of-domain high-error cases
# ------------------------------------------------------------

phase10_top_out_of_domain_errors = (
    phase10_predictions[
        phase10_predictions["combined_ad_class"].eq("out_of_domain")
    ]
    .sort_values(["target", "validation_regime", "abs_error"], ascending=[True, True, False])
    .groupby(["target", "protocol", "model", "validation_regime"], as_index=False)
    .head(30)
    .copy()
)


# ------------------------------------------------------------
# 12) Save outputs
# ------------------------------------------------------------

phase10_ad_scores.to_excel("phase10_ad_scores.xlsx", index=False)
phase10_predictions.to_excel("phase10_predictions_with_ad.xlsx", index=False)
phase10_error_by_ad_class.to_excel("phase10_error_by_ad_class.xlsx", index=False)
phase10_reliability_filtering_summary.to_excel("phase10_reliability_filtering_summary.xlsx", index=False)
phase10_ad_error_association.to_excel("phase10_ad_error_association.xlsx", index=False)
phase10_top_out_of_domain_errors.to_excel("phase10_top_out_of_domain_errors.xlsx", index=False)

print("\nSaved Phase 10 files:")
print("- phase10_ad_scores.xlsx")
print("- phase10_predictions_with_ad.xlsx")
print("- phase10_error_by_ad_class.xlsx")
print("- phase10_reliability_filtering_summary.xlsx")
print("- phase10_ad_error_association.xlsx")
print("- phase10_top_out_of_domain_errors.xlsx")


# ------------------------------------------------------------
# 13) Compact console preview
# ------------------------------------------------------------

print("\nAD class distribution and error summary:")
preview = phase10_error_by_ad_class[
    [
        "target",
        "protocol",
        "role",
        "validation_regime",
        "combined_ad_class",
        "n",
        "class_fraction",
        "mean_abs_error",
        "rmse",
    ]
].copy()

for col in ["class_fraction", "mean_abs_error", "rmse"]:
    preview[col] = preview[col].round(3)

print(
    preview.sort_values(
        ["target", "role", "validation_regime", "combined_ad_class"]
    ).to_string(index=False)
)

print("\nReliability filtering summary:")
filter_preview = phase10_reliability_filtering_summary[
    [
        "target",
        "protocol",
        "role",
        "validation_regime",
        "filter_name",
        "n_kept",
        "fraction_kept",
        "mean_abs_error",
        "rmse",
        "coverage_abs_90",
    ]
].copy()

for col in ["fraction_kept", "mean_abs_error", "rmse", "coverage_abs_90"]:
    filter_preview[col] = filter_preview[col].round(3)

print(
    filter_preview.sort_values(
        ["target", "role", "validation_regime", "filter_name"]
    ).to_string(index=False)
)

print("\nAD score-error association:")
assoc_preview = phase10_ad_error_association.copy()

for col in assoc_preview.select_dtypes(include=[np.number]).columns:
    assoc_preview[col] = assoc_preview[col].round(3)

print(assoc_preview.to_string(index=False))
