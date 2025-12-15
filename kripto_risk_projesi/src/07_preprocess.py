import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Veri yükleme, temizleme, aykırı değer temizleme, ölçeklendirme ve veri hazırlama fonksiyonları
def load_data(path):
    try:
        df = pd.read_csv(path)
        # Sütun isimlerinin kontrolü
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Sütun isimlendirme ve düzenleme
        rename_map = {
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close'
        }
        df.rename(columns=rename_map, inplace=True)
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
#Temizleme
def clean_data(df):
    if df.isnull().sum().sum() > 0:
        # Medyan ile sayısal stünları doldurma
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

#Aykırı değer temizleme
def remove_outliers_zscore(df, columns, threshold=3):
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < threshold]
    return df_clean

#Standartlaştırma 
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    # Sadece train seti üzerinden fit yapıyoruz, data leakage önlemek için
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # DataFrame'e geri çevir (okunabilirlik için)
    X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    
    return X_train_df, X_test_df, scaler

#Ölçeklendirme ve veri hazırlama
def prepare_data(df, target_col, feature_cols, test_size=0.2):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

#Train test ve validation
def prepare_data_with_val(df, target_col, feature_cols, test_size=0.2, val_size=0.2):
    # Özellik ve hedef ayrımı
    X = df[feature_cols]
    y = df[target_col]

    n = len(df)
    n_test = int(n * test_size)
    # Validation setini, test dışındaki kısımdan ayırma işlemi
    n_val = int((n - n_test) * val_size)

    # Zaman sırasını koruyarak bölünür
    X_train = X.iloc[: n - n_test - n_val]
    y_train = y.iloc[: n - n_test - n_val]

    X_val = X.iloc[n - n_test - n_val : n - n_test]
    y_val = y.iloc[n - n_test - n_val : n - n_test]

    X_test = X.iloc[n - n_test :]
    y_test = y.iloc[n - n_test :]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_val_df = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)
    X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, scaler