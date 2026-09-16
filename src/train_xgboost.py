import pandas as pd

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# Load raw downloaded data
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

df["volatility_20d"] = (
    df["return_1d"]
    .rolling(20)
    .std()
)

df["volume_ratio_20d"] = (
    df["Volume"]
    / df["Volume"].rolling(20).mean()
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

# Remove rows where we don't have enough history
# or don't know the next day's return
df = df.dropna(
    subset=[
        "return_1d",
        "return_5d",
        "volatility_20d",
        "volume_ratio_20d",
        "future_return_1d",
    ]
)

# -------------------------
# X and y
# -------------------------

features = [
    "return_1d",
    "return_5d",
    "volatility_20d",
    "volume_ratio_20d",
]

X = df[features]
y = df["target"]

# -------------------------
# Chronological split
# -------------------------

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]

# -------------------------
# Train baseline model
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
            )

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_accuracy = accuracy_score(y_train, train_pred)
            val_accuracy = accuracy_score(y_val, val_pred)

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

print("\nBest validation accuracy:", best_score)
print("Best parameters:", best_params)

validation_baseline = y_val.mean()
print("Validation always-up baseline:", validation_baseline)

# -------------------------
# Final model
# Retrain on train + validation
# -------------------------

X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

final_model = XGBClassifier(
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    n_estimators=best_params["n_estimators"],
    random_state=42,
)

final_model.fit(X_train_final, y_train_final)

# -------------------------
# Final holdout evaluation
# -------------------------

holdout_pred = final_model.predict(X_test)

holdout_accuracy = accuracy_score(
    y_test,
    holdout_pred,
)

holdout_baseline = y_test.mean()

print("\n--- Final Holdout Evaluation ---")
print("Best parameters:", best_params)
print("Holdout accuracy:", holdout_accuracy)
print("Always-up holdout baseline:", holdout_baseline)

print("Predicted up:", (holdout_pred == 1).sum())
print("Predicted down:", (holdout_pred == 0).sum())