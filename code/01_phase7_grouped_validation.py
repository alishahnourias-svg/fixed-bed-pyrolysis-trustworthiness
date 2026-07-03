# ============================================================
# Phase 7 - Validation Regimes, Revised Version
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
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_SPLITS = 5

DATA_PATH = Path("clean_audit_data_with_family.xlsx")
assert DATA_PATH.exists(), f"File not found: {DATA_PATH.resolve()}"

df = pd.read_excel(DATA_PATH)
df = df.reset_index(drop=True)
df["row_id"] = df.index.astype(int)
df["excel_row_number"] = df.index.astype(int) + 2

print("Loaded:", DATA_PATH)
print("Shape:", df.shape)
print("Columns:", len(df.columns))


REQUIRED_COLUMNS = [
    "row_id", "excel_row_number", "feedstock", "feedstock_family",
    "source_group", "source_pair_group", "biochar_yield", "bio_oil_yield",
    "carbon", "hydrogen", "nitrogen", "oxygen", "moisture",
    "volatile_matter", "fixed_carbon", "ash", "particle_size_mm",
    "temperature_c", "heating_rate_c_min", "gas_flow_rate_l_min",
    "residence_time_min", "raw_material_supply_g",
]

missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")
print("Required-column check passed.")


NUMERIC_FEATURES = [
    "carbon", "hydrogen", "nitrogen", "oxygen", "moisture",
    "volatile_matter", "fixed_carbon", "ash", "particle_size_mm",
    "temperature_c", "heating_rate_c_min", "gas_flow_rate_l_min",
    "residence_time_min", "raw_material_supply_g",
]

TARGETS = {
    "biochar": "biochar_yield",
    "bio_oil": "bio_oil_yield",
}

PROTOCOLS = {
    "A_numeric_only": {
        "numeric": NUMERIC_FEATURES,
        "categorical": [],
        "description": "Composition and operating variables only; no feedstock identity.",
    },
    "B_feedstock_identity": {
        "numeric": NUMERIC_FEATURES,
        "categorical": ["feedstock"],
        "description": "Composition and operating variables plus exact feedstock identity.",
    },
    "C_feedstock_family": {
        "numeric": NUMERIC_FEATURES,
        "categorical": ["feedstock_family"],
        "description": "Composition and operating variables plus feedstock-family taxonomy.",
    },
}

MODELS = {
    "Dummy_mean": DummyRegressor(strategy="mean"),
    "Ridge": Ridge(alpha=1.0),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.1,
    ),
}


def make_onehot_encoder():
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(protocol_name):
    if protocol_name not in PROTOCOLS:
        raise ValueError(f"Unknown protocol_name: {protocol_name}")

    protocol = PROTOCOLS[protocol_name]
    numeric_features = protocol["numeric"]
    categorical_features = protocol["categorical"]

    transformers = []

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    transformers.append(("numeric", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_transformer, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_pipeline(model, protocol_name):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(protocol_name)),
            ("model", clone(model)),
        ]
    )


def safe_group_string(series):
    return (
        series.astype("string")
        .fillna("MISSING")
        .str.replace(r"\.0$", "", regex=True)
    )


def get_model_data(dataframe, target_key, protocol_name):
    if target_key not in TARGETS:
        raise ValueError(f"Unknown target_key: {target_key}")
    if protocol_name not in PROTOCOLS:
        raise ValueError(f"Unknown protocol_name: {protocol_name}")

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


def get_cv_splits(X, y, meta, regime_name, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    if regime_name == "random_kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(X, y))

    group_key_map = {
        "source_group_kfold": "source_group",
        "feedstock_group_kfold": "feedstock",
        "family_group_kfold": "feedstock_family",
    }

    if regime_name not in group_key_map:
        raise ValueError(f"Unknown validation regime: {regime_name}")

    group_col = group_key_map[regime_name]
    group_values = meta[group_col]
    n_unique_groups = group_values.nunique()

    if n_unique_groups < 2:
        raise ValueError(f"Not enough unique groups for {regime_name}: {n_unique_groups}")

    effective_splits = min(n_splits, n_unique_groups)
    splitter = GroupKFold(n_splits=effective_splits)
    return list(splitter.split(X, y, groups=group_values))


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


def evaluate_model_cv(dataframe, target_key, protocol_name, model_name, model, regime_name):
    X, y, meta = get_model_data(
        dataframe=dataframe,
        target_key=target_key,
        protocol_name=protocol_name,
    )

    splits = get_cv_splits(
        X=X,
        y=y,
        meta=meta,
        regime_name=regime_name,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    prediction_rows = []

    group_key_for_regime = {
        "random_kfold": None,
        "source_group_kfold": "source_group",
        "feedstock_group_kfold": "feedstock",
        "family_group_kfold": "feedstock_family",
    }[regime_name]

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        meta_test = meta.iloc[test_idx].copy()

        pipe = make_model_pipeline(model=model, protocol_name=protocol_name)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)

        if group_key_for_regime is None:
            n_test_groups = np.nan
            test_groups_preview = ""
        else:
            test_group_values = meta_test[group_key_for_regime]
            n_test_groups = test_group_values.nunique()
            test_groups_preview = ";".join(sorted(test_group_values.unique())[:20])

        fold_rows.append({
            "target": target_key,
            "protocol": protocol_name,
            "model": model_name,
            "validation_regime": regime_name,
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_test_groups": n_test_groups,
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "median_ae": metrics["median_ae"],
            "test_groups_preview": test_groups_preview,
        })

        prediction_rows.append(pd.DataFrame({
            "target": target_key,
            "protocol": protocol_name,
            "model": model_name,
            "validation_regime": regime_name,
            "fold": fold_id,
            "row_id": meta_test["row_id"].values,
            "excel_row_number": meta_test["excel_row_number"].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "signed_error": y_pred - y_test.values,
            "abs_error": np.abs(y_pred - y_test.values),
            "source_group": meta_test["source_group"].values,
            "source_pair_group": meta_test["source_pair_group"].values,
            "feedstock": meta_test["feedstock"].values,
            "feedstock_family": meta_test["feedstock_family"].values,
        }))

    return pd.DataFrame(fold_rows), pd.concat(prediction_rows, ignore_index=True)


VALIDATION_REGIMES = [
    "random_kfold",
    "source_group_kfold",
    "feedstock_group_kfold",
    "family_group_kfold",
]

all_fold_results = []
all_prediction_results = []

total_runs = len(TARGETS) * len(PROTOCOLS) * len(MODELS) * len(VALIDATION_REGIMES)
run_counter = 0

for target_key in TARGETS:
    for protocol_name in PROTOCOLS:
        for model_name, model in MODELS.items():
            for regime_name in VALIDATION_REGIMES:
                run_counter += 1
                print(f"[{run_counter}/{total_runs}]", target_key, protocol_name, model_name, regime_name)

                fold_df, pred_df = evaluate_model_cv(
                    dataframe=df,
                    target_key=target_key,
                    protocol_name=protocol_name,
                    model_name=model_name,
                    model=model,
                    regime_name=regime_name,
                )
                all_fold_results.append(fold_df)
                all_prediction_results.append(pred_df)

fold_results = pd.concat(all_fold_results, ignore_index=True)
prediction_results = pd.concat(all_prediction_results, ignore_index=True)


fold_summary = (
    fold_results
    .groupby(["target", "protocol", "model", "validation_regime"], as_index=False)
    .agg(
        n_folds=("fold", "count"),
        total_test_rows=("n_test", "sum"),
        mean_r2=("r2", "mean"),
        std_r2=("r2", "std"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_median_ae=("median_ae", "mean"),
        std_median_ae=("median_ae", "std"),
    )
)


def pooled_metric_row(group):
    metrics = calculate_metrics(group["y_true"].values, group["y_pred"].values)
    return pd.Series({
        "pooled_n": len(group),
        "pooled_r2": metrics["r2"],
        "pooled_rmse": metrics["rmse"],
        "pooled_mae": metrics["mae"],
        "pooled_median_ae": metrics["median_ae"],
    })


try:
    pooled_summary = (
        prediction_results
        .groupby(["target", "protocol", "model", "validation_regime"])
        .apply(pooled_metric_row, include_groups=False)
        .reset_index()
    )
except TypeError:
    pooled_summary = (
        prediction_results
        .groupby(["target", "protocol", "model", "validation_regime"])
        .apply(pooled_metric_row)
        .reset_index()
    )


summary_results = fold_summary.merge(
    pooled_summary,
    on=["target", "protocol", "model", "validation_regime"],
    how="left",
)


random_baseline = (
    summary_results[summary_results["validation_regime"] == "random_kfold"]
    [[
        "target", "protocol", "model",
        "mean_r2", "mean_rmse", "mean_mae",
        "pooled_r2", "pooled_rmse", "pooled_mae",
    ]]
    .rename(columns={
        "mean_r2": "random_mean_r2",
        "mean_rmse": "random_mean_rmse",
        "mean_mae": "random_mean_mae",
        "pooled_r2": "random_pooled_r2",
        "pooled_rmse": "random_pooled_rmse",
        "pooled_mae": "random_pooled_mae",
    })
)

summary_results = summary_results.merge(
    random_baseline,
    on=["target", "protocol", "model"],
    how="left",
)

summary_results["mean_r2_drop_vs_random"] = summary_results["random_mean_r2"] - summary_results["mean_r2"]
summary_results["mean_rmse_increase_vs_random"] = summary_results["mean_rmse"] - summary_results["random_mean_rmse"]
summary_results["mean_mae_increase_vs_random"] = summary_results["mean_mae"] - summary_results["random_mean_mae"]
summary_results["pooled_r2_drop_vs_random"] = summary_results["random_pooled_r2"] - summary_results["pooled_r2"]
summary_results["pooled_rmse_increase_vs_random"] = summary_results["pooled_rmse"] - summary_results["random_pooled_rmse"]
summary_results["pooled_mae_increase_vs_random"] = summary_results["pooled_mae"] - summary_results["random_pooled_mae"]


key_cols_for_best = [
    "target", "validation_regime", "protocol", "model",
    "mean_r2", "mean_rmse", "mean_mae",
    "pooled_r2", "pooled_rmse", "pooled_mae",
    "mean_r2_drop_vs_random",
    "mean_rmse_increase_vs_random",
    "mean_mae_increase_vs_random",
    "pooled_r2_drop_vs_random",
    "pooled_rmse_increase_vs_random",
    "pooled_mae_increase_vs_random",
]

best_models_by_regime = (
    summary_results
    .sort_values(["target", "validation_regime", "mean_rmse"])
    .groupby(["target", "validation_regime"], as_index=False)
    .head(5)
    [key_cols_for_best]
    .copy()
)


fold_results.to_excel("phase7_v2_fold_results.xlsx", index=False)
prediction_results.to_excel("phase7_v2_prediction_results.xlsx", index=False)
summary_results.to_excel("phase7_v2_summary_results.xlsx", index=False)
best_models_by_regime.to_excel("phase7_v2_best_models_by_regime.xlsx", index=False)

print("\nSaved revised files:")
print("- phase7_v2_fold_results.xlsx")
print("- phase7_v2_prediction_results.xlsx")
print("- phase7_v2_summary_results.xlsx")
print("- phase7_v2_best_models_by_regime.xlsx")


preview_cols = [
    "target", "validation_regime", "protocol", "model",
    "mean_r2", "mean_rmse", "mean_mae",
    "pooled_r2", "pooled_rmse", "pooled_mae",
    "mean_r2_drop_vs_random", "pooled_r2_drop_vs_random",
]

preview = best_models_by_regime[preview_cols].copy()
numeric_cols = preview.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    preview[col] = preview[col].round(3)

print("\nBest models by target and validation regime:")
print(preview.to_string(index=False))
