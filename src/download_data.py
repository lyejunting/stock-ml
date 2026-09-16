import yfinance as yf


ticker = "AAPL"

df = yf.download(
    ticker,
    start="2015-01-01",
    end="2026-01-01",
    auto_adjust=False,
)

# Flatten yfinance MultiIndex columns for a single ticker
df.columns = df.columns.get_level_values(0)

# Feature engineering
df["return_1d"] = df["Adj Close"].pct_change()
df["return_5d"] = df["Adj Close"].pct_change(5)
df["volatility_20d"] = df["return_1d"].rolling(20).std()
df["volume_ratio_20d"] = df["Volume"] / df["Volume"].rolling(20).mean()

# Build the target: tomorrow's return
df["future_return_1d"] = df["Adj Close"].pct_change().shift(-1)

# Classification label:
# 1 = stock goes up tomorrow
# 0 = stock does not go up tomorrow
df["target"] = (df["future_return_1d"] > 0).astype(int)

# Remove rows where engineered features / target cannot be calculated
df = df.dropna(
    subset=[
        "return_1d",
        "return_5d",
        "volatility_20d",
        "volume_ratio_20d",
        "future_return_1d",
    ]
)

features = [
    "return_1d",
    "return_5d",
    "volatility_20d",
    "volume_ratio_20d",
]

X = df[features]
y = df["target"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

print("\nTrain period:")
print(X_train.index.min(), "to", X_train.index.max())

print("\nTest period:")
print(X_test.index.min(), "to", X_test.index.max())