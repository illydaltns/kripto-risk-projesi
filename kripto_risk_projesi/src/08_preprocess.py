import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = 'kripto_risk_projesi/data/ohlcv_clean.csv'
DESCRIBE_PATH = 'kripto_risk_projesi/outputs/describe_risk_volatility.csv' 


def load_data(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Hata: Dosya bulunamadı. '{path}' yolunu kontrol edin.")
        return None
    
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df

def fill_missing(df):
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
    return df

def load_describe_stats(path):
    try:
        stats = pd.read_csv(path, index_col=0)
        return stats
    except FileNotFoundError:
        print("Describe dosyası bulunamadı")
        return None

def remove_outliers(df, stats):
    if stats is None:
        return df
        
    df_copy = df.copy() # Orijinal dataframe'i korumak için kopya
    for col in stats.index:
        if col in df_copy.columns:
            mean = stats.loc[col, "mean"]
            std = stats.loc[col, "std"]
            lower = mean - 3 * std 
            upper = mean + 3 * std
            df_copy = df_copy[(df_copy[col] >= lower) & (df_copy[col] <= upper)]
    return df_copy

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    return X_scaled_df, scaler


df = load_data(DATA_PATH)
if df is None:
    exit()

df = fill_missing(df)

stats = load_describe_stats(DESCRIBE_PATH)
df = remove_outliers(df, stats)

features = ['volatility', 'momentum', 'pct_change', 'volume_btc', 'trade_count']
X = df[features]
y = df['risk']

X_scaled, scaler = scale_features(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False 
)

print(f"Ön İşleme Tamamlandı. Eğitim Veri Boyutu: {X_train.shape}")
print("-" * 40)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n{name} metrikleri:")
    print(f"R²  : {r2:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print("-" * 40)
    return y_pred, model, {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae}


print("\nLineer regresyon modeli")
lin_reg = LinearRegression()
lin_pred, lin_reg, lin_metrics = evaluate_model(
    "Lineer regresyon", lin_reg, X_train, y_train, X_test, y_test
)

print("\nRandom forest modeli")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_pred, rf_reg, rf_metrics = evaluate_model(
    "Random forest", rf_reg, X_train, y_train, X_test, y_test
)

residuals_lin = y_test - lin_pred
residuals_rf = y_test - rf_pred
print("\nKarşılaştırmalı görselleştirmeler")

# A. GERÇEK vs TAHMİN GRAFİĞİ
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lin_pred, color='blue', alpha=0.6, label='Lineer Reg.')
plt.scatter(y_test, rf_pred, color='orange', alpha=0.6, label='Random Forest')

# y=x mükemmel tahmin çizgisini çizme
min_val = min(y_test.min(), lin_pred.min(), rf_pred.min())
max_val = max(y_test.max(), lin_pred.max(), rf_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label=r'Mükemmel Tahmin ($y=\hat{y}$)')

plt.title('1. Gerçek risk vs tahmin edilen risk')
plt.xlabel(r'Gerçek Risk Değeri ($y$)')
plt.ylabel(r'Tahmin Edilen Risk Değeri ($\hat{y}$)') 
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()


# B. REZİDÜ (HATA) GRAFİĞİ
plt.figure(figsize=(10, 6))
plt.scatter(lin_pred, residuals_lin, color='green', alpha=0.6, label='Lineer Reg.')
plt.scatter(rf_pred, residuals_rf, color='purple', alpha=0.6, label='Random Forest')
plt.axhline(y=0, color='red', linestyle='--', label='Sıfır Hata Çizgisi') 
plt.title('2. Rezidü (Hata) Grafiği: Tahminler vs. Hatalar')
plt.xlabel(r'Tahmin Edilen Risk Değeri ($\hat{y}$)')
plt.ylabel(r'Hata (Rezidü): $y - \hat{y}$') 
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()


# C. REZİDÜ DAĞILIMI
plt.figure(figsize=(10, 6))
sns.histplot(residuals_lin, bins=30, kde=True, color='blue', stat='density', label='Lineer Reg.')
sns.histplot(residuals_rf, bins=30, kde=True, color='orange', stat='density', alpha=0.5, label='Random Forest')
plt.title('3. Rezidü (Hata) Dağılımı')
plt.xlabel('Hata (Rezidü) Değerleri')
plt.ylabel('Yoğunluk')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()


# D. KOEFİSYEN BÜYÜKLÜĞÜ GRAFİĞİ (Lineer regresyon)
coefs = lin_reg.coef_
feature_names = X.columns
coef_df = pd.DataFrame({'Özellik': feature_names, 'Katsayı': coefs})
coef_df['Mutlak Değer'] = coef_df['Katsayı'].abs()
coef_df = coef_df.sort_values(by='Mutlak Değer', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Katsayı', y='Özellik', data=coef_df, palette='viridis')
plt.title('4. Özellik Katsayıları (Lineer Regresyon)')
plt.xlabel('Katsayı')
plt.ylabel('Özellikler')
plt.grid(True, linestyle='--', alpha=0.3)
plt.axvline(0, color='black', linewidth=1)
plt.show()

# E. ÖZELLİK ÖNEMİ GRAFİĞİ (Random forest)
importances = rf_reg.feature_importances_
rf_feature_importance_df = pd.DataFrame({'Özellik': feature_names, 'Önem': importances})
rf_feature_importance_df = rf_feature_importance_df.sort_values(by='Önem', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Önem', y='Özellik', data=rf_feature_importance_df, palette='magma')
plt.title('5. Özellik Önem Grafiği (Random Forest)')
plt.xlabel('Önem Skoru')
plt.ylabel('Özellikler')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()