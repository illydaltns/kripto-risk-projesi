import requests
import pandas as pd
import os

SAVE_PATH = "C:/Users/Lenovo/Desktop/kripto_risk_projesi/data/ohlcv_fast.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

TOTAL_REQUESTS = 10
all_data = []

for i in range(TOTAL_REQUESTS):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "limit": 1000
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    all_data.extend(data)

columns = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume",
    "number_of_trades", "taker_buy_base", "taker_buy_quote",
    "ignore"
]

df = pd.DataFrame(all_data, columns=columns)

df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

numeric_cols = ["open", "high", "low", "close", "volume", "number_of_trades"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv(SAVE_PATH, index=False, encoding="utf-8")

print("Toplama işlemi tamamlandı.")
print(df.head())
