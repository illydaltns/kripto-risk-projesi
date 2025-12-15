import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Adds technical indicators to the dataframe:
    - SMA_14
    - RSI
    - Bollinger Bands (Middle)
    - Volatility (if missing)
    - Momentum (if missing)
    - Pct Change (if missing)
    """
    df = df.copy()
    
    # SMA 14
    if 'sma_14' not in df.columns:
        df['sma_14'] = df['close'].rolling(window=14).mean()
    
    # RSI 14
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands Middle (20 SMA)
    if 'bb_middle' not in df.columns:
        df['bb_middle'] = df['close'].rolling(window=20).mean()
    
    # Volatility (Rolling Std Dev of returns) - assuming 30 days window logic implies volatility of returns
    if 'volatility' not in df.columns:
        # pct_change might not exist yet
        returns = df['close'].pct_change()
        df['volatility'] = returns.rolling(window=30).std()
    
    # Momentum (10 days difference)
    if 'momentum' not in df.columns:
        df['momentum'] = df['close'].diff(10)

    # Percentage Change
    if 'pct_change' not in df.columns:
        df['pct_change'] = df['close'].pct_change()

    return df
