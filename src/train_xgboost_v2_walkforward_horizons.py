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
    df["Adj Close"] / df["ma_10"] - 1
)

df["distance_ma_20"] = (
    df["Adj Close"] / df["ma_20"] - 1
)

df["high_low_range"] = (
    (df["High"] - df["Low"])
    / df["Adj Close"]
)


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


# Remove rows that do not yet have
# enough history for our features
df = df.dropna(
    subset=features
).copy()


# -------------------------
# Experiment setup
# -------------------------

horizons = [
    1,
    5,
    20,
]

validation_years = [
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
]


# Keep hyperparameters fixed initially.
#
# This lets us isolate the effect of changing
# the prediction horizon.
#
# Later, we can tune each horizon independently.
params = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "random_state": 42,
    "eval_metric": "logloss",
}


all_results = []


# =========================================================
# Horizon loop
# =========================================================

for horizon in horizons:

    print("\n")
    print("==========================================")
    print(f"Prediction horizon: {horizon} trading days")
    print("==========================================")


    # -----------------------------------------------------
    # Target
    #
    # Example for horizon=5:
    #
    # Today:
    #     price = $100
    #
    # Five trading days later:
    #     price = $105
    #
    # future_return = +5%
    # target = 1
    # -----------------------------------------------------

    target_return_column = (
        f"future_return_{horizon}d"
    )

    target_column = (
        f"target_{horizon}d"
    )


    df[target_return_column] = (
        df["Adj Close"]
        .pct_change(horizon)
        .shift(-horizon)
    )

    df[target_column] = (
        df[target_return_column] > 0
    ).astype(int)


    # Dataset specifically for this horizon.
    #
    # Last N rows cannot have labels because
    # we don't know what happens N days later.
    horizon_df = df.dropna(
        subset=[
            target_return_column
        ]
    ).copy()

    X = horizon_df[features]
    y = horizon_df[target_column]


    horizon_results = []


    # =====================================================
    # Walk-forward validation
    # =====================================================

    for validation_year in validation_years:

        validation_mask = (
            X.index.year == validation_year
        )

        X_val = X.loc[validation_mask]
        y_val = y.loc[validation_mask]


        if len(X_val) == 0:
            continue


        validation_start = (
            X_val.index.min()
        )


        # Everything before validation year is
        # initially eligible for training.
        train_mask = (
            X.index < validation_start
        )

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]


        # -------------------------------------------------
        # PURGE
        #
        # Extremely important for future-return targets.
        #
        # With a 20-day target, the last 20 training
        # rows could use prices from the validation
        # period to create their labels.
        #
        # Remove those rows.
        # -------------------------------------------------

        if len(X_train) > horizon:

            X_train = (
                X_train.iloc[:-horizon]
            )

            y_train = (
                y_train.iloc[:-horizon]
            )


        model = XGBClassifier(
            **params
        )

        model.fit(
            X_train,
            y_train,
        )


        train_pred = (
            model.predict(X_train)
        )

        val_pred = (
            model.predict(X_val)
        )


        train_accuracy = (
            accuracy_score(
                y_train,
                train_pred,
            )
        )

        val_accuracy = (
            accuracy_score(
                y_val,
                val_pred,
            )
        )


        # Majority / always-up baseline
        always_up_baseline = (
            y_val.mean()
        )


        model_minus_baseline = (
            val_accuracy
            - always_up_baseline
        )


        result = {
            "horizon": horizon,
            "validation_year": validation_year,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "always_up_baseline": always_up_baseline,
            "model_minus_baseline": model_minus_baseline,
        }


        horizon_results.append(
            result
        )

        all_results.append(
            result
        )


        print(
            f"\nValidation year: "
            f"{validation_year}"
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


    # =====================================================
    # Horizon summary
    # =====================================================

    horizon_results_df = (
        pd.DataFrame(
            horizon_results
        )
    )


    print("\n------------------------------------------")
    print(
        f"{horizon}-day horizon summary"
    )
    print("------------------------------------------")


    print(
        horizon_results_df[
            [
                "validation_year",
                "validation_accuracy",
                "always_up_baseline",
                "model_minus_baseline",
            ]
        ].to_string(
            index=False
        )
    )


    average_accuracy = (
        horizon_results_df[
            "validation_accuracy"
        ].mean()
    )

    average_baseline = (
        horizon_results_df[
            "always_up_baseline"
        ].mean()
    )

    average_edge = (
        horizon_results_df[
            "model_minus_baseline"
        ].mean()
    )

    years_beating_baseline = (
        (
            horizon_results_df[
                "model_minus_baseline"
            ] > 0
        ).sum()
    )


    print(
        "\nAverage model accuracy:",
        average_accuracy,
    )

    print(
        "Average always-up baseline:",
        average_baseline,
    )

    print(
        "Average model - baseline:",
        average_edge,
    )

    print(
        "Years model beat baseline:",
        years_beating_baseline,
        "/",
        len(horizon_results_df),
    )


    # =====================================================
    # SHAP
    #
    # Explain the most recent year
    # for this prediction horizon.
    # =====================================================

    shap_year = (
        validation_years[-1]
    )


    shap_mask = (
        X.index.year == shap_year
    )

    X_shap = (
        X.loc[shap_mask]
    )

    y_shap = (
        y.loc[shap_mask]
    )


    if len(X_shap) == 0:
        continue


    shap_start = (
        X_shap.index.min()
    )


    shap_train_mask = (
        X.index < shap_start
    )


    X_train_shap = (
        X.loc[shap_train_mask]
    )

    y_train_shap = (
        y.loc[shap_train_mask]
    )


    # Purge again to prevent target leakage
    if len(X_train_shap) > horizon:

        X_train_shap = (
            X_train_shap.iloc[:-horizon]
        )

        y_train_shap = (
            y_train_shap.iloc[:-horizon]
        )


    shap_model = (
        XGBClassifier(
            **params
        )
    )

    shap_model.fit(
        X_train_shap,
        y_train_shap,
    )


    shap_pred = (
        shap_model.predict(
            X_shap
        )
    )


    shap_accuracy = (
        accuracy_score(
            y_shap,
            shap_pred,
        )
    )

    shap_baseline = (
        y_shap.mean()
    )


    # -------------------------
    # SHAP values
    # -------------------------

    explainer = (
        shap.TreeExplainer(
            shap_model,
            feature_perturbation=(
                "tree_path_dependent"
            ),
        )
    )


    shap_values = (
        explainer(X_shap)
    )


    # -------------------------
    # Beeswarm
    # -------------------------

    shap.plots.beeswarm(
        shap_values,
        show=False,
    )


    plt.tight_layout()


    shap_filename = (
        f"shap_horizon_"
        f"{horizon}d_"
        f"{shap_year}.png"
    )


    plt.savefig(
        shap_filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    print(
        f"\nSHAP year: "
        f"{shap_year}"
    )

    print(
        f"SHAP model accuracy: "
        f"{shap_accuracy:.4f}"
    )

    print(
        f"SHAP always-up baseline: "
        f"{shap_baseline:.4f}"
    )

    print(
        f"Saved SHAP beeswarm to "
        f"{shap_filename}"
    )


# =========================================================
# Compare horizons
# =========================================================

all_results_df = (
    pd.DataFrame(
        all_results
    )
)


horizon_summary = (
    all_results_df
    .groupby("horizon")
    .agg(
        average_accuracy=(
            "validation_accuracy",
            "mean",
        ),
        average_baseline=(
            "always_up_baseline",
            "mean",
        ),
        average_edge=(
            "model_minus_baseline",
            "mean",
        ),
    )
)


horizon_summary[
    "years_beating_baseline"
] = (
    all_results_df
    .assign(
        beat_baseline=(
            all_results_df[
                "model_minus_baseline"
            ] > 0
        )
    )
    .groupby("horizon")[
        "beat_baseline"
    ]
    .sum()
)


print("\n")
print("==========================================")
print("FINAL HORIZON COMPARISON")
print("==========================================")

print(
    horizon_summary
    .to_string()
)