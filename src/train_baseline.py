import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# -------------------------
# Train baseline model
# -------------------------

model = LogisticRegression()

model.fit(X_train, y_train)

# -------------------------
# Predict
# -------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(y_prob[:10])

# -------------------------
# Evaluate
# -------------------------

accuracy = accuracy_score(y_test, y_pred)

always_up_baseline = y_test.mean()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print()
print("Model accuracy:", accuracy)
print("Always-up baseline:", always_up_baseline)

print("Predicted up:", (y_pred == 1).sum())
print("Predicted down:", (y_pred == 0).sum())

print("\nCoefficients:")
for feature, coef in zip(features, model.coef_[0]):
    print(feature, coef)

print("\nIntercept:")
print(model.intercept_[0])