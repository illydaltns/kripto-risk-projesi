import pandas as pd
import numpy as np


def add_technical_indicators(df):
    df = df.copy()

    # 1. Volatility (High - Low)
    df["volatility"] = df["high"] - df["low"]

    # 2. Momentum (Close - Open)
    df["momentum"] = df["close"] - df["open"]

    # 3. Percent Change
    df["pct_change"] = df["close"].pct_change()

    # 4. Simple Moving Average (SMA) - 14 günlük
    df["sma_14"] = df["close"].rolling(window=14).mean()

    # 5. RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 6. Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_upper"] = df["bb_middle"] + 2 * df["close"].rolling(window=20).std()
    df["bb_lower"] = df["bb_middle"] - 2 * df["close"].rolling(window=20).std()

    # Rolling yüzünden oluşan nan değerleri doldurulur
    df = df.bfill()

    return df

