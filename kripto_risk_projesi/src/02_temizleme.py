import pandas as pd
import os

FILE_PATH = "C:/Users/Lenovo/Desktop/kripto_risk_projesi/data/ohlcv_fast.csv"
df = pd.read_csv(FILE_PATH)

print("Eksik veri sayısı:")
print(df.isna().sum())

drop_cols = [
    "quote_asset_volume",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore"
]

df = df.drop(columns=drop_cols, errors="ignore")

SAVE_PATH = "C:/Users/Lenovo/Desktop/kripto_risk_projesi/data/ohlcv_clean.csv"
df.to_csv(SAVE_PATH, index=False, encoding="utf-8")

print("Temiz dosya kaydedildi.")
