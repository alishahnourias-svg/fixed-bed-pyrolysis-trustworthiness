# %%
import inspect
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor


# ------------------------------------------------------------
# 1) Load cleaned audit data
# ------------------------------------------------------------

DATA_PATH = Path("clean_audit_data.xlsx")

assert DATA_PATH.exists(), f"File not found: {DATA_PATH.resolve()}"

df = pd.read_excel(DATA_PATH)

print("Loaded:", DATA_PATH)
print("Shape:", df.shape)
print("Columns:", len(df.columns))


# ------------------------------------------------------------
# 2) Define feedstock-family mapping
# ------------------------------------------------------------

FEEDSTOCK_FAMILY_MAP = {
    # Woody biomass
    "Wood": "woody_biomass",
    "Pine": "woody_biomass",
    "Bamboo": "woody_biomass",
    "Oak": "woody_biomass",

    # Agricultural residues and grasses
    "Rice Husk": "agricultural_residue",
    "Corncob": "agricultural_residue",
    "Wheat Straw": "agricultural_residue",
    "Oat Straw": "agricultural_residue",
    "Cotton Stalk": "agricultural_residue",
    "Rapeseed Stalk": "agricultural_residue",
    "Sugarcane Bagasse": "agricultural_residue",
    "Switchgrass": "agricultural_residue",
    "Miscanthus": "agricultural_residue",
    "Timothy Grass": "agricultural_residue",
    "Reed": "agricultural_residue",
    "Sesame Stalk": "agricultural_residue",
    "Garlic Stem": "agricultural_residue",
    "Pepper Stem": "agricultural_residue",
    "Lemon Grass": "agricultural_residue",
    "Tea Waste": "agricultural_residue",
    "Casava": "agricultural_residue",
    "Palm Fiber": "agricultural_residue",
    "Tobacco Residues": "agricultural_residue",
    "Jerusalem Artichoke Stick": "agricultural_residue",
    "Jute Dust": "agricultural_residue",
    "Laurel Residues": "agricultural_residue",

    # Shells, husks, kernels, hard residues
    "Peanut Shell": "shell_husk_kernel",
    "Walnut Shell": "shell_husk_kernel",
    "Palm Shell": "shell_husk_kernel",
    "Coconut Waste": "shell_husk_kernel",
    "Apricot Kernel Shell": "shell_husk_kernel",
    "Hazelnut Cupula": "shell_husk_kernel",
    "Hornbeam Shell": "shell_husk_kernel",
    "Cherry Seed": "shell_husk_kernel",
    "Pomegranate Seeds": "shell_husk_kernel",

    # Manure, litter, sludge
    "Manure": "manure_sludge",
    "Sewage Sludge": "manure_sludge",
    "Poultry Litter": "manure_sludge",
    "Oil Sludge": "manure_sludge",

    # Algae
    "Algae": "algae",

    # Oilseed, seed cake, and seed-derived residues
    "Jatropha Cake": "oilseed_residue",
    "Olive Kernel": "oilseed_residue",
    "Milk Thistle Seeds": "oilseed_residue",
    "Pistacia Lentiscus Seeds": "oilseed_residue",
    "Linseed": "oilseed_residue",
    "Babool Seed": "oilseed_residue",
    "Safflower Seed": "oilseed_residue",
    "Soybean Cake": "oilseed_residue",
    "Kasaud Seed": "oilseed_residue",
    "Pongamia Seed": "oilseed_residue",
    "Mesua seed": "oilseed_residue",
    "Calophyllum Inophyllum": "oilseed_residue",
    "Mahua Seed": "oilseed_residue",
    "Karanja Seed": "oilseed_residue",
    "Sunflower Bagasse": "oilseed_residue",

    # Food and fruit processing residues
    "Food Waste": "food_fruit_waste",
    "Olive Husk": "food_fruit_waste",
    "Orange Pomace": "food_fruit_waste",
    "Grape Bagasse": "food_fruit_waste",
    "Lychee": "food_fruit_waste",
    "Tomato Peel": "food_fruit_waste",

    # Pure or semi-pure biochemical components
    "Cellulose": "pure_component",
   "Hemicellulose(Xylan)": "pure_component",
   "Hemicellulose(xylan)": "pure_component",
    "Lignin": "pure_component",

    # Other plant materials / low-frequency botanical residues
    "Euphorbia Rigida": "other_biomass",
    "Eremurus Spectabilis": "other_biomass",
    "Ferula Orientalis L.": "other_biomass",
    "Avocado Seed": "other_biomass",
    "Liquorice": "other_biomass",
}


df["feedstock_family"] = df["feedstock"].map(FEEDSTOCK_FAMILY_MAP)

unmapped = sorted(df.loc[df["feedstock_family"].isna(), "feedstock"].dropna().unique())

print("\nUnique feedstocks:", df["feedstock"].nunique())
print("Mapped feedstocks:", df.loc[df["feedstock_family"].notna(), "feedstock"].nunique())
print("Unmapped feedstocks:", len(unmapped))

if unmapped:
    print("\nUnmapped feedstocks. Please check manually:")
    for item in unmapped:
        print("-", item)
else:
    print("\nAll feedstocks were mapped to a family.")

# Use explicit label only after printing unmapped items
df["feedstock_family"] = df["feedstock_family"].fillna("other_unmapped")


# ------------------------------------------------------------
# 3) Define allowed features
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
        "description": "Composition and operating variables plus broader feedstock-family taxonomy.",
    },
}


# ------------------------------------------------------------
# 4) One-hot encoder compatible with different sklearn versions
# ------------------------------------------------------------

def make_onehot_encoder():
    params = inspect.signature(OneHotEncoder).parameters

    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ------------------------------------------------------------
# 5) Leakage-safe preprocessing pipeline
# ------------------------------------------------------------

def make_preprocessor(protocol_name):
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

    transformers.append(
        ("numeric", numeric_transformer, numeric_features)
    )

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_encoder()),
            ]
        )

        transformers.append(
            ("categorical", categorical_transformer, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def make_pipeline(model, protocol_name):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(protocol_name)),
            ("model", model),
        ]
    )


# ------------------------------------------------------------
# 6) Function to prepare X, y, and group labels
# ------------------------------------------------------------

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

    groups = {
        "source_group": dataframe.loc[rows, "source_group"].astype(str).copy(),
        "source_pair_group": dataframe.loc[rows, "source_pair_group"].astype(str).copy(),
        "feedstock": dataframe.loc[rows, "feedstock"].astype(str).copy(),
        "feedstock_family": dataframe.loc[rows, "feedstock_family"].astype(str).copy(),
    }

    return X, y, groups


# ------------------------------------------------------------
# 7) Smoke test: check that all protocols can be fitted
#    This is NOT model evaluation.
# ------------------------------------------------------------

for target_key in TARGETS:
    print(f"\nTarget: {target_key}")

    for protocol_name in PROTOCOLS:
        X, y, groups = get_model_data(df, target_key, protocol_name)

        pipe = make_pipeline(
            model=DummyRegressor(strategy="mean"),
            protocol_name=protocol_name
        )

        # This fit is only a technical test that the pipeline works.
        # Do not report this as model performance.
        pipe.fit(X, y)

        transformed = pipe.named_steps["preprocess"].transform(X.head(5))

        print(
            protocol_name,
            "| X shape:", X.shape,
            "| y:", y.shape,
            "| transformed preview shape:", transformed.shape,
            "| source groups:", groups["source_group"].nunique(),
            "| feedstocks:", groups["feedstock"].nunique(),
            "| families:", groups["feedstock_family"].nunique(),
        )


# ------------------------------------------------------------
# 8) Save data with feedstock-family labels
# ------------------------------------------------------------

df.to_excel("clean_audit_data_with_family.xlsx", index=False)

df[df["biochar_yield"].notna()].to_excel(
    "biochar_dataset_with_family.xlsx", index=False
)

df[df["bio_oil_yield"].notna()].to_excel(
    "bio_oil_dataset_with_family.xlsx", index=False
)

df[df["biochar_yield"].notna() & df["bio_oil_yield"].notna()].to_excel(
    "paired_dataset_with_family.xlsx", index=False
)

print("\nProtocol-ready datasets saved successfully.")
