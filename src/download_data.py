import yfinance as yf

ticker = "AAPL"

df = yf.download(
    ticker,
    start="2015-01-01",
    end="2026-01-01",
    auto_adjust=False,
)

df.columns = df.columns.get_level_values(0)

df.to_csv("data/aapl.csv")