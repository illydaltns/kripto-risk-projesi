import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# DOSYA YOLLARI
# --------------------------------------------------
DATA_PATH = 'kripto_risk_projesi/data/ohlcv_clean.csv'
DESCRIBE_PATH = 'kripto_risk_projesi/outputs/describe_risk_volatility.csv'


# --------------------------------------------------
# VERİ YÜKLEME
# --------------------------------------------------
def load_data(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Hata: Dosya bulunamadı -> {path}")
        return None

    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df


# --------------------------------------------------
# EKSİK VERİ DOLDURMA
# --------------------------------------------------
def fill_missing(df):
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
    return df


# --------------------------------------------------
# DESCRIBE İSTATİSTİKLERİ
# --------------------------------------------------
def load_describe_stats(path):
    try:
        stats = pd.read_csv(path, index_col=0)
        return stats
    except FileNotFoundError:
        print("Describe dosyası bulunamadı.")
        return None


# --------------------------------------------------
# AYKIRI DEĞER TEMİZLEME (3 SIGMA)
# --------------------------------------------------
def remove_outliers(df, stats):
    if stats is None:
        return df

    df_copy = df.copy()

    for col in stats.index:
        if col in df_copy.columns:
            mean = stats.loc[col, "mean"]
            std = stats.loc[col, "std"]
            lower = mean - 3 * std
            upper = mean + 3 * std

            df_copy = df_copy[
                (df_copy[col] >= lower) &
                (df_copy[col] <= upper)
            ]

    return df_copy


# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(
        X_scaled,
        index=X.index,
        columns=X.columns
    )

    return X_scaled_df, scaler


# --------------------------------------------------
# MODEL DEĞERLENDİRME
# --------------------------------------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{name} metrikleri:")
    print(f"R²   : {r2:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print("-" * 40)

    return y_pred, model


# --------------------------------------------------
# MAIN PIPELINE HELPERS
# --------------------------------------------------
def clean_data(df):
    """
    Genel veri temizliği.
    """
    df = df.copy()
    df = fill_missing(df)
    return df

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Z-score yöntemi ile outlier temizliği.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            if std > 0:
                z_scores = np.abs((df_clean[col] - mean) / std)
                df_clean = df_clean[z_scores < threshold]
    return df_clean

def prepare_data(df, target_col, feature_cols, test_size=0.2):
    """
    Train/Test split ve scaling.
    """
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def prepare_data_with_val(df, target_col, feature_cols, test_size=0.2, val_size=0.25):
    """
    Train/Validation/Test split ve scaling.
    """
    X = df[feature_cols]
    y = df[target_col]
    
    # Test ayrımı
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Validation ayrımı
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


# ==================================================
# ANA AKIŞ
# ==================================================
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    if df is not None:
        df = fill_missing(df)

        stats = load_describe_stats(DESCRIBE_PATH)
        df = remove_outliers(df, stats)

        features = [
            'volatility',
            'momentum',
            'pct_change',
            'volume_btc',
            'trade_count'
        ]

        # Check if features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Missing features in dataframe: {missing_features}")
        else:
            X = df[features]
            y = df['risk']

            X_scaled, scaler = scale_features(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                y,
                test_size=0.2,
                shuffle=False
            )

            print(f"Eğitim veri boyutu: {X_train.shape}")
            print("-" * 40)


            # --------------------------------------------------
            # MODELLER
            # --------------------------------------------------
            print("\nLineer Regresyon")
            lin_reg = LinearRegression()
            lin_pred, lin_reg = evaluate_model(
                "Lineer Regresyon",
                lin_reg,
                X_train,
                y_train,
                X_test,
                y_test
            )

            print("\nRandom Forest")
            rf_reg = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            rf_pred, rf_reg = evaluate_model(
                "Random Forest",
                rf_reg,
                X_train,
                y_train,
                X_test,
                y_test
            )



            # --------------------------------------------------
            # GÖRSELLEŞTİRMELER
            # --------------------------------------------------
            print("\nKarşılaştırmalı görselleştirmeler")

            # A) GERÇEK vs TAHMİN
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, lin_pred, alpha=0.6, label='Lineer Regresyon')
            plt.scatter(y_test, rf_pred, alpha=0.6, label='Random Forest')

            min_val = min(y_test.min(), lin_pred.min(), rf_pred.min())
            max_val = max(y_test.max(), lin_pred.max(), rf_pred.max())

            plt.plot([min_val, max_val], [min_val, max_val], '--', label='y = ŷ')
            plt.xlabel("Gerçek Risk")
            plt.ylabel("Tahmin Edilen Risk")
            plt.title("Gerçek vs Tahmin")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()


            # B) REZİDÜ GRAFİĞİ
            residuals_lin = y_test - lin_pred
            residuals_rf = y_test - rf_pred

            plt.figure(figsize=(10, 6))
            plt.scatter(lin_pred, residuals_lin, alpha=0.6, label='Lineer Regresyon')
            plt.scatter(rf_pred, residuals_rf, alpha=0.6, label='Random Forest')
            plt.axhline(0, linestyle='--')
            plt.xlabel("Tahmin")
            plt.ylabel("Hata (Rezidü)")
            plt.title("Rezidü Grafiği")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()


            # C) REZİDÜ DAĞILIMI
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals_lin, bins=30, kde=True, stat='density', label='Lineer')
            sns.histplot(residuals_rf, bins=30, kde=True, stat='density', alpha=0.5, label='RF')
            plt.title("Rezidü Dağılımı")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()


            # D) LINEER REGRESYON KATSAYILARI
            coef_df = pd.DataFrame({
                'Özellik': X.columns,
                'Katsayı': lin_reg.coef_
            })
            coef_df['Mutlak'] = coef_df['Katsayı'].abs()
            coef_df = coef_df.sort_values(by='Mutlak', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(
                x='Katsayı',
                y='Özellik',
                data=coef_df,
                hue='Özellik',
                palette='viridis',
                legend=False
            )
            plt.axvline(0)
            plt.title("Lineer Regresyon Katsayıları")
            plt.grid(alpha=0.3)
            plt.show()


            # E) RANDOM FOREST FEATURE IMPORTANCE
            rf_imp_df = pd.DataFrame({
                'Özellik': X.columns,
                'Önem': rf_reg.feature_importances_
            })
            if not rf_imp_df.empty:
                 rf_imp_df = rf_imp_df.sort_values(by='Önem', ascending=False)
                 
                 plt.figure(figsize=(10, 6))
                 sns.barplot(
                    x='Önem',
                    y='Özellik',
                    data=rf_imp_df,
                    hue='Özellik',
                    palette='magma',
                    legend=False
                 )
                 plt.title("Random Forest Özellik Önemi")
                 plt.grid(alpha=0.3)
                 plt.show()
