import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# -------------------------
# Load data
# -------------------------

df = pd.read_csv(
    "data/aapl.csv",
    index_col="Date",
    parse_dates=True,
)


# -------------------------
# Feature engineering
# -------------------------

df["return_1d"] = df["Adj Close"].pct_change()
df["return_5d"] = df["Adj Close"].pct_change(5)
df["return_10d"] = df["Adj Close"].pct_change(10)
df["return_20d"] = df["Adj Close"].pct_change(20)

df["volatility_20d"] = (
    df["return_1d"]
    .rolling(20)
    .std()
)

df["volume_ratio_20d"] = (
    df["Volume"]
    / df["Volume"].rolling(20).mean()
)

df["ma_10"] = (
    df["Adj Close"]
    .rolling(10)
    .mean()
)

df["ma_20"] = (
    df["Adj Close"]
    .rolling(20)
    .mean()
)

df["distance_ma_10"] = (
    df["Adj Close"]
    / df["ma_10"]
    - 1
)

df["distance_ma_20"] = (
    df["Adj Close"]
    / df["ma_20"]
    - 1
)

df["high_low_range"] = (
    (df["High"] - df["Low"])
    / df["Adj Close"]
)


# -------------------------
# Target
# -------------------------

df["future_return_1d"] = (
    df["Adj Close"]
    .pct_change()
    .shift(-1)
)

df["target"] = (
    df["future_return_1d"] > 0
).astype(int)


# -------------------------
# Features
# -------------------------

features = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_20d",
    "volume_ratio_20d",
    "distance_ma_10",
    "distance_ma_20",
    "high_low_range",
]

df = df.dropna(
    subset=features + ["future_return_1d"]
)

X = df[features]
y = df["target"]


# -------------------------
# Walk-forward setup
# -------------------------

validation_years = [
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
]

results = []

# Reuse our best params from v2
params = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "random_state": 42,
    "eval_metric": "logloss",
}


# -------------------------
# Walk-forward loop
# -------------------------

for validation_year in validation_years:

    train_mask = X.index.year < validation_year
    val_mask = X.index.year == validation_year

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]

    if len(X_val) == 0:
        continue

    model = XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
    )

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_accuracy = accuracy_score(
        y_train,
        train_pred,
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred,
    )

    always_up_baseline = y_val.mean()

    model_minus_baseline = (
        val_accuracy - always_up_baseline
    )

    results.append(
        {
            "validation_year": validation_year,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "always_up_baseline": always_up_baseline,
            "model_minus_baseline": model_minus_baseline,
        }
    )

    print(
        f"\nValidation year: {validation_year}"
    )

    print(
        f"Train period: "
        f"{X_train.index.min().date()} "
        f"to "
        f"{X_train.index.max().date()}"
    )

    print(
        f"Validation period: "
        f"{X_val.index.min().date()} "
        f"to "
        f"{X_val.index.max().date()}"
    )

    print(
        f"Train accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Always-up baseline: "
        f"{always_up_baseline:.4f}"
    )

    print(
        f"Model - baseline: "
        f"{model_minus_baseline:.4f}"
    )


# -------------------------
# Walk-forward summary
# -------------------------

results_df = pd.DataFrame(results)

print("\n==============================")
print("Walk-forward summary")
print("==============================")

print(
    results_df[
        [
            "validation_year",
            "validation_accuracy",
            "always_up_baseline",
            "model_minus_baseline",
        ]
    ].to_string(index=False)
)

print("\nAverage model accuracy:")
print(
    results_df["validation_accuracy"].mean()
)

print("\nAverage always-up baseline:")
print(
    results_df["always_up_baseline"].mean()
)

print("\nAverage model - baseline:")
print(
    results_df["model_minus_baseline"].mean()
)

print("\nYears model beat baseline:")
print(
    (
        results_df["model_minus_baseline"] > 0
    ).sum(),
    "/",
    len(results_df),
)


# -------------------------
# SHAP beeswarm
#
# Train using all history before
# the most recent validation year,
# then explain that year's predictions.
# -------------------------

shap_year = validation_years[-1]

train_mask = X.index.year < shap_year
shap_mask = X.index.year == shap_year

X_train_shap = X.loc[train_mask]
y_train_shap = y.loc[train_mask]

X_shap = X.loc[shap_mask]
y_shap = y.loc[shap_mask]


# Train model specifically for SHAP
final_shap_model = XGBClassifier(**params)

final_shap_model.fit(
    X_train_shap,
    y_train_shap,
)


# -------------------------
# SHAP values
# -------------------------

explainer = shap.TreeExplainer(
    final_shap_model,
    feature_perturbation="tree_path_dependent",
)

shap_values = explainer(X_shap)


# -------------------------
# SHAP beeswarm plot
# -------------------------

shap.plots.beeswarm(
    shap_values,
    show=False,
)

plt.tight_layout()

plt.savefig(
    "shap_walkforward_beeswarm.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    "\nSaved SHAP beeswarm to "
    "shap_walkforward_beeswarm.png"
)


# -------------------------
# SHAP model performance
# for the explained year
# -------------------------

shap_pred = final_shap_model.predict(X_shap)

shap_accuracy = accuracy_score(
    y_shap,
    shap_pred,
)

shap_baseline = y_shap.mean()

print(
    f"\nSHAP year: {shap_year}"
)

print(
    f"SHAP year model accuracy: "
    f"{shap_accuracy:.4f}"
)

print(
    f"SHAP year always-up baseline: "
    f"{shap_baseline:.4f}"
)