import pandas as pd

FILE_PATH = "C:/Users/Lenovo/Desktop/kripto_risk_projesi/data/ohlcv_clean.csv"
df = pd.read_csv(FILE_PATH)

df["volatility"] = df["high"] - df["low"]
df["momentum"] = df["close"] - df["open"]
df["percent_change"] = (df["close"] - df["open"]) / df["open"]
df["risk_score"] = (df["high"] - df["low"]) / df["volume"]

SAVE_PATH = "C:/Users/Lenovo/Desktop/kripto_risk_projesi/data/ohlcv_features.csv"
df.to_csv(SAVE_PATH, index=False, encoding="utf-8")

print("Yeni özellikler eklendi.")
print(df.head())
