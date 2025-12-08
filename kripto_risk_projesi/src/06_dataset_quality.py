import pandas as pd

df = pd.read_csv("C:/Users/Lenovo/Desktop/kripto_risk_projesi/data/ohlcv_clean.csv")

print("Veri Seti Boyutu:", df.shape)
print("\nVeri Türleri:")
print(df.dtypes)
print("\nEksik Veri Kontrolü:")
print(df.isnull().sum())
print("\nSütunlar:")
print(df.columns.tolist())
