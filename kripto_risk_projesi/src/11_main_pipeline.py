import sys
import os

# Mevcut dizini path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import importlib.util

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

current_dir = os.path.dirname(__file__)

# 10_technical_indicators
ti = import_module_from_path('technical_indicators', os.path.join(current_dir, '10_technical_indicators.py'))
add_technical_indicators = ti.add_technical_indicators
# 07_preprocess
pp = import_module_from_path('preprocess', os.path.join(current_dir, '07_preprocess.py'))
load_data = pp.load_data
clean_data = pp.clean_data
remove_outliers_zscore = pp.remove_outliers_zscore
prepare_data = pp.prepare_data
prepare_data_with_val = pp.prepare_data_with_val
# 09_regression_model
rm = import_module_from_path('regression_model', os.path.join(current_dir, '09_regression_model.py'))
train_linear_regression = rm.train_linear_regression
train_random_forest = rm.train_random_forest
hyperparameter_tuning = rm.hyperparameter_tuning
# 08_regression_eval
re = import_module_from_path('regression_eval', os.path.join(current_dir, '08_regression_eval.py'))
calculate_metrics = re.calculate_metrics
plot_actual_vs_predicted = re.plot_actual_vs_predicted
plot_residuals = re.plot_residuals
plot_feature_importance = re.plot_feature_importance
plot_correlation_matrix = re.plot_correlation_matrix
plot_residual_distribution = re.plot_residual_distribution
plot_learning_curve = re.plot_learning_curve
plot_model_comparison = re.plot_model_comparison
get_linear_regression_coefficients = re.get_linear_regression_coefficients

# Sabitler
DATA_PATH = 'kripto_risk_projesi/data/ohlcv_clean.csv'

def main():
    print("1. Veri Yükleniyor...")
    # Resole absolute path to data file
    # file is in ../data/ohlcv_clean.csv relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, '..', 'data', 'ohlcv_clean.csv')
    path = os.path.abspath(path)
    
    print(f"Veri yolu: {path}")
    
    df = load_data(path)
    print(f"Veri yüklendi: {df.shape}")
    print(f"Sütunlar: {df.columns.tolist()}")
    
    # Standardize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    
    # Rename columns to standard names if needed
    rename_map = {
        'open_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'close_price': 'close',
        'volume_btc': 'volume'
    }
    df.rename(columns=rename_map, inplace=True)
    
    print(f"Sütunlar (standart): {df.columns.tolist()}")

    print("\n2. Teknik İndikatörler Ekleniyor...")
    df = add_technical_indicators(df)
    print(f"Yeni özellikler eklendi: {df.shape}")

    print("\n3. Veri Temizliği ve Preprocessing...")
    df = clean_data(df)
    # Target 'risk' oluştur (Eğer veri setinde yoksa, eski koddaki logic: (high-low)/volume)
    if 'risk' not in df.columns:
         # Eski kodda risk_score vardı, onu kullanalım veya yeniden hesaplayalım
         if 'risk_score' in df.columns:
             df['risk'] = df['risk_score']
         else:
             df['risk'] = (df['high'] - df['low']) / df['volume']
             
    # Outlier temizliği (Sayısal sütunlar için)
    numeric_cols = ['volatility', 'momentum', 'pct_change', 'rsi']
    df = remove_outliers_zscore(df, numeric_cols)
    print(f"Outlier temizliği sonrası: {df.shape}")

    print("\n4. Veri Hazırlığı (Train / Validation / Test Split & Scale)...")
    feature_cols = ['volatility', 'momentum', 'pct_change', 'sma_14', 'rsi', 'bb_middle']
    # NaN varsa tekrar temizle (indicatorlerden gelenler)
    df.dropna(inplace=True)
    
    # Görselleştirme 1: Korelasyon Matrisi
    print("\n4.0. Korelasyon Matrisi Çiziliyor...")
    plot_correlation_matrix(df, feature_cols, 'risk')

    # Zaman sıralı Train / Validation / Test ayrımı + ölçekleme
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler = prepare_data_with_val(
        df,
        target_col='risk',
        feature_cols=feature_cols,
        test_size=0.2,   # Son %20: test
        val_size=0.25    # Kalanın %25'i: validation -> yaklaşık %60 / %20 / %20
    )
    print(f"Train boyutu: {X_train_scaled.shape}, Validation boyutu: {X_val_scaled.shape}, Test boyutu: {X_test_scaled.shape}")

    metrics_dict = {}

    print("\n5. Model Eğitimi (Linear Regression)...")
    lr_model = train_linear_regression(X_train_scaled, y_train)
    # Validation seti performansı
    y_val_pred_lr = lr_model.predict(X_val_scaled)
    _ = calculate_metrics(y_val, y_val_pred_lr, "Linear Regression (Validation)")

    # Test seti performansı
    y_pred_lr = lr_model.predict(X_test_scaled)
    metrics_dict["Linear Regression"] = calculate_metrics(y_test, y_pred_lr, "Linear Regression (Test)")
    # Katsayı tablosu (açıklanabilirlik)
    lr_coef_df = get_linear_regression_coefficients(lr_model, feature_cols)
    print("\nLinear Regression Katsayıları (büyükten küçüğe |coef|):")
    print(lr_coef_df)
    
    # Görselleştirme: Actual vs Predicted ve Residuals
    print("\n5.1. Linear Regression Görselleştirmeleri...")
    plot_actual_vs_predicted(y_test, y_pred_lr, "Linear Regression")
    plot_residuals(y_test, y_pred_lr, "Linear Regression")
    plot_residual_distribution(y_test, y_pred_lr, "Linear Regression")
    plot_learning_curve(lr_model, X_train_scaled, y_train, "Linear Regression")

    print("\n6. Model Eğitimi (Random Forest + TimeSeries Cross-Validation)...")
    # TimeSeriesSplit ile hiperparametre ayarı
    rf_model, rf_best_params = hyperparameter_tuning(X_train_scaled, y_train, n_splits=5)
    print(f"Random Forest (CV) en iyi parametreler: {rf_best_params}")

    # Validation seti performansı
    y_val_pred_rf = rf_model.predict(X_val_scaled)
    _ = calculate_metrics(y_val, y_val_pred_rf, "Random Forest (Validation)")

    # Test seti performansı
    y_pred_rf = rf_model.predict(X_test_scaled)
    metrics_dict["Random Forest (CV)"] = calculate_metrics(y_test, y_pred_rf, "Random Forest (Test)")
    
    # Görselleştirme: Actual vs Predicted, Residuals ve Feature Importance
    print("\n6.1. Random Forest Görselleştirmeleri...")
    plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest")
    plot_residuals(y_test, y_pred_rf, "Random Forest")
    plot_feature_importance(rf_model, feature_cols)
    plot_residual_distribution(y_test, y_pred_rf, "Random Forest")
    plot_learning_curve(rf_model, X_train_scaled, y_train, "Random Forest")

    # Görselleştirme: Model Karşılaştırma
    print("\n7. Model Karşılaştırması...")
    plot_model_comparison(metrics_dict)

main()
