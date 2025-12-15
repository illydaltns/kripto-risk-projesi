import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import learning_curve

#performans metriklerini hesaplama ve görselleştirme fonksiyonları
def calculate_metrics(y_true, y_pred, model_name="Model"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics = {
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    }
    
    print(f"___ {model_name} Performansı ___")
    print(f"R²  : {r2:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print("-" * 30)
    
    return metrics

#Gerçek değer ve tahmin edilen değer karşılaştırılması
def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Mükemmel Tahmin')
    
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Edilen Değerler')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

#Rezidü 
def plot_residuals(y_true, y_pred, title="Residual Plot"):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Tahmin Edilen Değerler')
    plt.ylabel('Hata (Rezidü)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

#Random Forest için özellik önem düzeylerini görselleştirme
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
        plt.title('Özellik Önem Düzeyleri')
        plt.xlabel('Önem Skoru')
        plt.ylabel('Özellikler')
        plt.show()
    else:
        print("Bu model için feature_importances_ özelliği bulunmuyor.")
#Korelasyon matrisi 
def plot_correlation_matrix(df, feature_cols, target_col='risk'):
    cols = feature_cols + [target_col]
    # Sadece dataframe'de var olan sütunlar alınır
    cols = [c for c in cols if c in df.columns]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Özellik Korelasyon Matrisi")
    plt.show()

#histogram ve kde
def plot_residual_distribution(y_true, y_pred, model_name="Model"):
    # Verileri numpy array'e çevirerek ve boyut uyumsuzluklarını önlenmeye çalışılır
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    residuals = y_true - y_pred
    
    # Sonsuz ve NaNs değerleri temizlenir
    mask = np.isfinite(residuals)
    if not np.all(mask):
        print(f"Uyarı: {model_name} residuals içinde {np.sum(~mask)} adet geçersiz (NaN/Inf) değer var. Bunlar temizleniyor.")
        residuals = residuals[mask]
        
    if len(residuals) == 0:
        print(f"Uyarı: {model_name} için çizilecek geçerli residual kalmadı.")
        return

    # Çok aşırı uç değerleri (outliers) görselleştirmede görmezden gelmek gerekebilir
    lower_bound = np.percentile(residuals, 0.5)
    upper_bound = np.percentile(residuals, 99.5)
    residuals_clipped = residuals[(residuals >= lower_bound) & (residuals <= upper_bound)]

    print(f"{model_name} Residuals - Min: {residuals_clipped.min()}, Max: {residuals_clipped.max()}, Mean: {residuals_clipped.mean()}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals_clipped, kde=True, color='purple', bins=50)
    plt.title(f"{model_name} - Hata (Residual) Dağılımı (Outliers Clipped)")
    plt.xlabel("Hata")
    plt.ylabel("Frekans")
    plt.axvline(x=0, color='red', linestyle='--')
    plt.show()

#Öğrenme eğrisi
def plot_learning_curve(model, X, y, model_name="Model"):
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='neg_mean_squared_error'
        )
        
        # MSE negatif döner, pozitife çevirip karekök alınır (RMSE)
        train_scores_mean = np.sqrt(-np.mean(train_scores, axis=1))
        test_scores_mean = np.sqrt(-np.mean(test_scores, axis=1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Eğitim Hatası (RMSE)")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Doğrulama Hatası (RMSE)")
        
        plt.title(f"{model_name} - Öğrenme Eğrisi")
        plt.xlabel("Eğitim Örnek Sayısı")
        plt.ylabel("RMSE")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Öğrenme eğrisi çizilirken hata oluştu: {e}")

#Model karşılaştırma ve bar grafiği ile görselleştirilmesi
def plot_model_comparison(metrics_dict):
    models = list(metrics_dict.keys())
    
    rmse_scores = [metrics_dict[m]['RMSE'] for m in models]
    r2_scores = [metrics_dict[m]['R2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, rmse_scores, width, label='RMSE (Düşük İyi)')
    rects2 = ax.bar(x + width/2, r2_scores, width, label='R2 (Yüksek İyi)')
    
    ax.set_ylabel('Skorlar')
    ax.set_title('Model Karşılaştırması')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Değerler barların üzerine yazılır
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    
    fig.tight_layout()
    plt.show()

# Lineer regresyon katsayılarını tablo olarak döndürür ve katsayılara göre mutlak değeri büyükten küçüğe sıralanır
def get_linear_regression_coefficients(model, feature_names):
    coefs = np.ravel(model.coef_)
    df = pd.DataFrame({
        "Özellik": feature_names,
        "Katsayı": coefs
    })
    df["|Katsayı|"] = df["Katsayı"].abs()
    return df.sort_values("|Katsayı|", ascending=False)