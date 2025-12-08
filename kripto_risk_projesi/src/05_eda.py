import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Lenovo/Desktop/kripto_risk_projesi/data/ohlcv_clean.csv")

df.head()
df.tail()
df.sample()
df.shape
df.info()
df.dtypes
df.isnull().sum()
df.describe()

df_numeric = df.select_dtypes(include=["float64", "int64"])
sns.heatmap(df_numeric.corr(), cmap="coolwarm")
plt.show()

sns.histplot(df["close_price"])
plt.show()

sns.histplot(df["volume_btc"])
plt.show()

sns.boxplot(df["risk"])
plt.show()

sns.boxplot(df["volatility"])
plt.show()

df["time"] = pd.to_datetime(df["time"])
df.set_index("time")["close_price"].plot(figsize=(12, 4))
plt.show()

sns.scatterplot(data=df, x="volatility", y="risk")
plt.show()
