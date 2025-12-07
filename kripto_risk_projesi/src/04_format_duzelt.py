import pandas as pd

PATH = "C:/Users/Lenovo/Desktop/kripto_risk_projesi/data/ohlcv_features.csv"
df = pd.read_csv(PATH)

df = df.rename(columns={
    "open_time": "time",
    "open": "open_price",
    "high": "high_price",
    "low": "low_price",
    "close": "close_price",
    "volume": "volume_btc",
    "number_of_trades": "trade_count",
    "percent_change": "pct_change",
    "risk_score": "risk"
})

df = df[
    [
        "time",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume_btc",
        "trade_count",
        "volatility",
        "momentum",
        "pct_change",
        "risk"
    ]
]

df["time"] = pd.to_datetime(df["time"])

float_cols = [
    "open_price", "high_price", "low_price", "close_price",
    "volume_btc", "trade_count", "volatility", "momentum",
    "pct_change", "risk"
]

for col in float_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").round(6)

SAVE = "C:/Users/Lenovo/Desktop/kripto_risk_projesi/data/ohlcv_clean.csv"

df.to_csv(
    SAVE,
    index=False,
    encoding="utf-8",
    decimal=".",
    float_format="%.6f"
)

print(df.head())
print(SAVE)
