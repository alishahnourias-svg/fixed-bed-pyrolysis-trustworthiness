# ============================================================
# Phase 9 - Uncertainty Calibration with Leakage-Safe Conformal Intervals
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

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.ensemble import ExtraTreesRegressor


# ------------------------------------------------------------
# 0) General settings
# ------------------------------------------------------------

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
CALIBRATION_SIZE = 0.20

# Nominal coverage levels for conformal prediction intervals
NOMINAL_LEVELS = [0.80, 0.90, 0.95]


# ------------------------------------------------------------
# 1) Load protocol-ready data
# ------------------------------------------------------------

DATA_PATH = Path("clean_audit_data_with_family.xlsx")
assert DATA_PATH.exists(), f"File not found: {DATA_PATH.resolve()}"

df = pd.read_excel(DATA_PATH)
df = df.reset_index(drop=True)
df["row_id"] = df.index.astype(int)
df["excel_row_number"] = df.index.astype(int) + 2

print("Loaded:", DATA_PATH)
print("Shape:", df.shape)
print("Columns:", len(df.columns))


# ------------------------------------------------------------
# 2) Feature protocols and selected Phase-8 model decisions
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

# Phase-8 official decisions:
# - Biochar: ExtraTrees + numeric-only protocol
# - Bio-oil: ExtraTrees + feedstock-family protocol as primary reliability-screening model
# - Bio-oil: ExtraTrees + numeric-only protocol as sensitivity model
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
# 3) Required-column check
# ------------------------------------------------------------

required_cols = [
    "feedstock",
    "feedstock_family",
    "source_group",
    "source_pair_group",
    "biochar_yield",
    "bio_oil_yield",
] + NUMERIC_FEATURES

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("Required-column check passed.")


# ------------------------------------------------------------
# 4) Preprocessing and model construction
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

        transformers.append(
            ("categorical", categorical_transformer, protocol["categorical"])
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_extra_trees_pipeline(protocol_name):
    model = ExtraTreesRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=1,              # n_jobs=1 avoids noisy parallel warnings in notebooks
        min_samples_leaf=2,
    )

    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(protocol_name)),
            ("model", model),
        ]
    )


# ------------------------------------------------------------
# 5) Utility functions
# ------------------------------------------------------------

def safe_group_string(series):
    return (
        series
        .astype("string")
        .fillna("MISSING")
        .str.replace(r"\.0$", "", regex=True)
    )


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual = y_pred - y_true

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def conformal_quantile(scores, alpha):
    """
    Finite-sample conformal quantile.
    alpha = 1 - nominal_coverage.
    """
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


def get_outer_cv_splits(X, y, meta, regime_name):
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


def split_outer_train_into_proper_and_calibration(
    outer_train_indices,
    y,
    meta,
    regime_name,
):
    """
    Splits the outer-training set into:
    - proper-training subset
    - calibration subset

    For grouped validation regimes, the inner calibration split also respects
    the corresponding grouping variable when enough groups are available.
    """
    outer_train_indices = np.asarray(outer_train_indices)

    if regime_name == "random_kfold":
        proper_idx, calib_idx = train_test_split(
            outer_train_indices,
            test_size=CALIBRATION_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        return proper_idx, calib_idx, "random_calibration_split"

    group_map = {
        "source_group_kfold": "source_group",
        "feedstock_group_kfold": "feedstock",
        "family_group_kfold": "feedstock_family",
    }

    group_col = group_map[regime_name]
    train_groups = meta.iloc[outer_train_indices][group_col].reset_index(drop=True)
    local_positions = np.arange(len(outer_train_indices))

    n_unique_groups = train_groups.nunique()

    if n_unique_groups >= 3:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=CALIBRATION_SIZE,
            random_state=RANDOM_STATE,
        )

        local_proper, local_calib = next(
            splitter.split(local_positions, groups=train_groups)
        )

        proper_idx = outer_train_indices[local_proper]
        calib_idx = outer_train_indices[local_calib]

        return proper_idx, calib_idx, f"grouped_calibration_split_by_{group_col}"

    # Conservative fallback: still only uses outer training data, so no outer-test leakage.
    # But it does not preserve the group structure inside the calibration split.
    proper_idx, calib_idx = train_test_split(
        outer_train_indices,
        test_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return proper_idx, calib_idx, "fallback_random_calibration_split_due_to_few_groups"


def predict_extra_trees_mean_std(pipe, X):
    """
    Returns:
    - mean prediction from the fitted ExtraTrees pipeline
    - standard deviation across individual trees as an ensemble uncertainty proxy
    """
    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    X_transformed = preprocessor.transform(X)

    tree_preds = np.column_stack([
        tree.predict(X_transformed) for tree in model.estimators_
    ])

    mean_pred = np.mean(tree_preds, axis=1)
    std_pred = np.std(tree_preds, axis=1, ddof=1)

    return mean_pred, std_pred


# ------------------------------------------------------------
# 6) Main conformal evaluation function
# ------------------------------------------------------------

def evaluate_conformal_for_spec(target_key, protocol_name, role, claim_strength, regime_name):
    X, y, meta = get_model_data(df, target_key, protocol_name)
    outer_splits = get_outer_cv_splits(X, y, meta, regime_name)

    prediction_rows = []
    fold_rows = []

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
        proper_idx, calib_idx, calibration_split_type = (
            split_outer_train_into_proper_and_calibration(
                outer_train_indices=outer_train_idx,
                y=y,
                meta=meta,
                regime_name=regime_name,
            )
        )

        X_proper = X.iloc[proper_idx].copy()
        y_proper = y.iloc[proper_idx].copy()

        X_calib = X.iloc[calib_idx].copy()
        y_calib = y.iloc[calib_idx].copy()

        X_test = X.iloc[outer_test_idx].copy()
        y_test = y.iloc[outer_test_idx].copy()
        meta_test = meta.iloc[outer_test_idx].copy()

        pipe = make_extra_trees_pipeline(protocol_name)
        pipe.fit(X_proper, y_proper)

        y_calib_pred, calib_std = predict_extra_trees_mean_std(pipe, X_calib)
        y_test_pred, test_std = predict_extra_trees_mean_std(pipe, X_test)

        calib_abs_resid = np.abs(y_calib.values - y_calib_pred)

        # Absolute split conformal scores
        absolute_scores = calib_abs_resid

        # Normalized conformal scores using ensemble standard deviation
        # The floor avoids unrealistically narrow intervals when tree variance is near zero.
        std_floor = np.nanquantile(calib_std, 0.10)
        if not np.isfinite(std_floor) or std_floor <= 0:
            std_floor = 1e-6

        calib_scale = np.maximum(calib_std, std_floor)
        test_scale = np.maximum(test_std, std_floor)

        normalized_scores = calib_abs_resid / calib_scale

        base_metrics = calculate_metrics(y_test, y_test_pred)

        fold_record = {
            "target": target_key,
            "protocol": protocol_name,
            "model": "ExtraTrees",
            "role": role,
            "claim_strength": claim_strength,
            "validation_regime": regime_name,
            "fold": fold_id,
            "n_proper_train": len(proper_idx),
            "n_calibration": len(calib_idx),
            "n_test": len(outer_test_idx),
            "calibration_split_type": calibration_split_type,
            "std_floor": std_floor,
            "r2": base_metrics["r2"],
            "rmse": base_metrics["rmse"],
            "mae": base_metrics["mae"],
            "median_ae": base_metrics["median_ae"],
        }

        for nominal in NOMINAL_LEVELS:
            alpha = 1 - nominal

            q_abs = conformal_quantile(absolute_scores, alpha=alpha)
            q_norm = conformal_quantile(normalized_scores, alpha=alpha)

            lower_abs = y_test_pred - q_abs
            upper_abs = y_test_pred + q_abs

            lower_norm = y_test_pred - q_norm * test_scale
            upper_norm = y_test_pred + q_norm * test_scale

            covered_abs = (y_test.values >= lower_abs) & (y_test.values <= upper_abs)
            covered_norm = (y_test.values >= lower_norm) & (y_test.values <= upper_norm)

            fold_record[f"coverage_abs_{int(nominal*100)}"] = np.mean(covered_abs)
            fold_record[f"width_abs_{int(nominal*100)}"] = np.mean(upper_abs - lower_abs)
            fold_record[f"q_abs_{int(nominal*100)}"] = q_abs

            fold_record[f"coverage_norm_{int(nominal*100)}"] = np.mean(covered_norm)
            fold_record[f"width_norm_{int(nominal*100)}"] = np.mean(upper_norm - lower_norm)
            fold_record[f"q_norm_{int(nominal*100)}"] = q_norm

            pred_tmp = pd.DataFrame({
                "target": target_key,
                "protocol": protocol_name,
                "model": "ExtraTrees",
                "role": role,
                "claim_strength": claim_strength,
                "validation_regime": regime_name,
                "fold": fold_id,
                "nominal_coverage": nominal,
                "row_id": meta_test["row_id"].values,
                "excel_row_number": meta_test["excel_row_number"].values,
                "y_true": y_test.values,
                "y_pred": y_test_pred,
                "signed_error": y_test_pred - y_test.values,
                "abs_error": np.abs(y_test_pred - y_test.values),
                "ensemble_std": test_std,
                "scale_used": test_scale,
                "lower_abs_conformal": lower_abs,
                "upper_abs_conformal": upper_abs,
                "covered_abs_conformal": covered_abs.astype(int),
                "width_abs_conformal": upper_abs - lower_abs,
                "lower_norm_conformal": lower_norm,
                "upper_norm_conformal": upper_norm,
                "covered_norm_conformal": covered_norm.astype(int),
                "width_norm_conformal": upper_norm - lower_norm,
                "source_group": meta_test["source_group"].values,
                "source_pair_group": meta_test["source_pair_group"].values,
                "feedstock": meta_test["feedstock"].values,
                "feedstock_family": meta_test["feedstock_family"].values,
                "calibration_split_type": calibration_split_type,
            })

            prediction_rows.append(pred_tmp)

        fold_rows.append(fold_record)

    return pd.DataFrame(fold_rows), pd.concat(prediction_rows, ignore_index=True)


# ------------------------------------------------------------
# 7) Run Phase 9
# ------------------------------------------------------------

all_fold_rows = []
all_prediction_rows = []

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

        fold_df, pred_df = evaluate_conformal_for_spec(
            target_key=spec["target"],
            protocol_name=spec["protocol"],
            role=spec["role"],
            claim_strength=spec["claim_strength"],
            regime_name=regime,
        )

        all_fold_rows.append(fold_df)
        all_prediction_rows.append(pred_df)

phase9_fold_results = pd.concat(all_fold_rows, ignore_index=True)
phase9_prediction_intervals = pd.concat(all_prediction_rows, ignore_index=True)


# ------------------------------------------------------------
# 8) Calibration summary
# ------------------------------------------------------------

summary_rows = []

group_cols = [
    "target",
    "protocol",
    "model",
    "role",
    "claim_strength",
    "validation_regime",
    "nominal_coverage",
]

for keys, group in phase9_prediction_intervals.groupby(group_cols):
    row = dict(zip(group_cols, keys))

    metrics = calculate_metrics(group["y_true"].values, group["y_pred"].values)

    row.update({
        "n_predictions": len(group),
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "median_ae": metrics["median_ae"],

        "coverage_abs_conformal": group["covered_abs_conformal"].mean(),
        "mean_width_abs_conformal": group["width_abs_conformal"].mean(),
        "median_width_abs_conformal": group["width_abs_conformal"].median(),

        "coverage_norm_conformal": group["covered_norm_conformal"].mean(),
        "mean_width_norm_conformal": group["width_norm_conformal"].mean(),
        "median_width_norm_conformal": group["width_norm_conformal"].median(),

        "mean_ensemble_std": group["ensemble_std"].mean(),
        "median_ensemble_std": group["ensemble_std"].median(),
        "mean_abs_error": group["abs_error"].mean(),
        "median_abs_error": group["abs_error"].median(),
    })

    summary_rows.append(row)

phase9_calibration_summary = pd.DataFrame(summary_rows)


# ------------------------------------------------------------
# 9) Error-uncertainty association diagnostics
# ------------------------------------------------------------

association_rows = []

base_group_cols = [
    "target",
    "protocol",
    "model",
    "role",
    "validation_regime",
]

# Use only one row per prediction for association diagnostics.
# The same prediction appears once per nominal coverage level, so keep nominal=0.90.
assoc_data = phase9_prediction_intervals[
    np.isclose(phase9_prediction_intervals["nominal_coverage"], 0.90)
].copy()

for keys, group in assoc_data.groupby(base_group_cols):
    row = dict(zip(base_group_cols, keys))

    if group["ensemble_std"].nunique() > 1 and group["abs_error"].nunique() > 1:
        pearson_corr = group[["ensemble_std", "abs_error"]].corr(method="pearson").iloc[0, 1]
        spearman_corr = group[["ensemble_std", "abs_error"]].corr(method="spearman").iloc[0, 1]
    else:
        pearson_corr = np.nan
        spearman_corr = np.nan

    row.update({
        "n": len(group),
        "pearson_corr_ensemble_std_abs_error": pearson_corr,
        "spearman_corr_ensemble_std_abs_error": spearman_corr,
        "mean_abs_error": group["abs_error"].mean(),
        "mean_ensemble_std": group["ensemble_std"].mean(),
    })

    association_rows.append(row)

phase9_error_uncertainty_association = pd.DataFrame(association_rows)


# ------------------------------------------------------------
# 10) Error and coverage by uncertainty bins
# ------------------------------------------------------------

bin_rows = []

for keys, group in assoc_data.groupby(base_group_cols):
    group = group.copy()

    # Use quantile bins. If duplicate bin edges occur, drop duplicates.
    try:
        group["uncertainty_bin"] = pd.qcut(
            group["ensemble_std"],
            q=5,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        continue

    for bin_id, bin_group in group.groupby("uncertainty_bin"):
        row = dict(zip(base_group_cols, keys))

        row.update({
            "uncertainty_bin": int(bin_id),
            "n": len(bin_group),
            "mean_ensemble_std": bin_group["ensemble_std"].mean(),
            "mean_abs_error": bin_group["abs_error"].mean(),
            "median_abs_error": bin_group["abs_error"].median(),
            "coverage_abs_90": bin_group["covered_abs_conformal"].mean(),
            "coverage_norm_90": bin_group["covered_norm_conformal"].mean(),
            "mean_width_abs_90": bin_group["width_abs_conformal"].mean(),
            "mean_width_norm_90": bin_group["width_norm_conformal"].mean(),
        })

        bin_rows.append(row)

phase9_uncertainty_bin_diagnostics = pd.DataFrame(bin_rows)


# ------------------------------------------------------------
# 11) Save outputs
# ------------------------------------------------------------

phase9_fold_results.to_excel("phase9_fold_results.xlsx", index=False)
phase9_prediction_intervals.to_excel("phase9_prediction_intervals.xlsx", index=False)
phase9_calibration_summary.to_excel("phase9_calibration_summary.xlsx", index=False)
phase9_error_uncertainty_association.to_excel("phase9_error_uncertainty_association.xlsx", index=False)
phase9_uncertainty_bin_diagnostics.to_excel("phase9_uncertainty_bin_diagnostics.xlsx", index=False)

print("\nSaved Phase 9 files:")
print("- phase9_fold_results.xlsx")
print("- phase9_prediction_intervals.xlsx")
print("- phase9_calibration_summary.xlsx")
print("- phase9_error_uncertainty_association.xlsx")
print("- phase9_uncertainty_bin_diagnostics.xlsx")


# ------------------------------------------------------------
# 12) Compact console preview
# ------------------------------------------------------------

preview_cols = [
    "target",
    "protocol",
    "role",
    "validation_regime",
    "nominal_coverage",
    "r2",
    "rmse",
    "mae",
    "coverage_abs_conformal",
    "mean_width_abs_conformal",
    "coverage_norm_conformal",
    "mean_width_norm_conformal",
]

preview = phase9_calibration_summary[preview_cols].copy()

for col in [
    "nominal_coverage",
    "r2",
    "rmse",
    "mae",
    "coverage_abs_conformal",
    "mean_width_abs_conformal",
    "coverage_norm_conformal",
    "mean_width_norm_conformal",
]:
    preview[col] = preview[col].round(3)

print("\nCalibration summary preview:")
print(
    preview.sort_values(
        ["target", "role", "validation_regime", "nominal_coverage"]
    ).to_string(index=False)
)

print("\nError-uncertainty association preview:")
assoc_preview = phase9_error_uncertainty_association.copy()

for col in [
    "pearson_corr_ensemble_std_abs_error",
    "spearman_corr_ensemble_std_abs_error",
    "mean_abs_error",
    "mean_ensemble_std",
]:
    assoc_preview[col] = assoc_preview[col].round(3)

print(assoc_preview.to_string(index=False))
