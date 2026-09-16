import pandas as pd

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
# Feature engineering v2
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

df["ma_10"] = df["Adj Close"].rolling(10).mean()
df["ma_20"] = df["Adj Close"].rolling(20).mean()

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
# Feature list
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

# Remove rows where any feature or target is unavailable
df = df.dropna(
    subset=features + ["future_return_1d"]
)

# -------------------------
# X / y
# -------------------------

X = df[features]
y = df["target"]

# -------------------------
# Chronological split
# 70% train
# 15% validation
# 15% holdout
# -------------------------

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Holdout:", X_test.shape)

print("\nTrain period:")
print(X_train.index.min(), "to", X_train.index.max())

print("\nValidation period:")
print(X_val.index.min(), "to", X_val.index.max())

print("\nHoldout period:")
print(X_test.index.min(), "to", X_test.index.max())

# -------------------------
# Hyperparameter search
# -------------------------

depths = [2, 3, 4, 5]
learning_rates = [0.01, 0.05, 0.1]
n_estimators_options = [50, 100, 200]

best_score = 0
best_params = None

for depth in depths:
    for learning_rate in learning_rates:
        for n_estimators in n_estimators_options:

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=depth,
                learning_rate=learning_rate,
                random_state=42,
                eval_metric="logloss",
            )

            model.fit(X_train, y_train)

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

            print(
                f"depth={depth} | "
                f"lr={learning_rate} | "
                f"trees={n_estimators} | "
                f"train={train_accuracy:.4f} | "
                f"validation={val_accuracy:.4f}"
            )

            if val_accuracy > best_score:
                best_score = val_accuracy
                best_params = {
                    "max_depth": depth,
                    "learning_rate": learning_rate,
                    "n_estimators": n_estimators,
                }

# -------------------------
# Validation summary
# -------------------------

validation_baseline = y_val.mean()

print("\n--- Validation Summary ---")
print("Best validation accuracy:", best_score)
print("Best parameters:", best_params)
print(
    "Validation always-up baseline:",
    validation_baseline,
)

# -------------------------
# Final training
# Train + validation
# -------------------------

X_train_final = pd.concat(
    [X_train, X_val]
)

y_train_final = pd.concat(
    [y_train, y_val]
)

final_model = XGBClassifier(
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    n_estimators=best_params["n_estimators"],
    random_state=42,
    eval_metric="logloss",
)

final_model.fit(
    X_train_final,
    y_train_final,
)

# -------------------------
# Final holdout evaluation
# -------------------------

holdout_pred = final_model.predict(X_test)
holdout_prob = final_model.predict_proba(X_test)[:, 1]

holdout_accuracy = accuracy_score(
    y_test,
    holdout_pred,
)

holdout_baseline = y_test.mean()

print("\n--- Final Holdout Evaluation ---")
print("Best parameters:", best_params)
print("Holdout accuracy:", holdout_accuracy)
print(
    "Always-up holdout baseline:",
    holdout_baseline,
)

print(
    "Predicted up:",
    (holdout_pred == 1).sum(),
)

print(
    "Predicted down:",
    (holdout_pred == 0).sum(),
)

print("\nFirst 10 probabilities:")
print(holdout_prob[:10])

# -------------------------
# Feature importance
# -------------------------

print("\n--- Feature Importance ---")

for feature, importance in sorted(
    zip(features, final_model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
):
    print(
        f"{feature}: {importance:.4f}"
    )