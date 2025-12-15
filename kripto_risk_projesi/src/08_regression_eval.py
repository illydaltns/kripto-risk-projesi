import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve

def calculate_metrics(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} Metrikler ---")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_residuals(y_true, y_pred, title="Residuals"):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Tahmin Edilen Değerler")
    plt.ylabel("Residü (Hata)")
    plt.title(f"{title} - Residual Plot")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_residual_distribution(y_true, y_pred, title="Residual Distribution"):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"{title} - Residual Distribution")
    plt.xlabel("Residü")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not have feature_importances_ attribute.")

def plot_correlation_matrix(df, feature_cols, target_col):
    cols = feature_cols + [target_col]
    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelasyon Matrisi")
    plt.show()

def plot_learning_curve(model, X, y, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='neg_mean_squared_error'
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score (MSE)")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score (MSE)")
    plt.title(f"{title} - Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("MSE (Lower is better)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_model_comparison(metrics_dict):
    # metrics_dict: {'ModelName': {'mse': val, 'rmse': val ...}}
    models = list(metrics_dict.keys())
    rmse_values = [m['rmse'] for m in metrics_dict.values()]
    r2_values = [m['r2'] for m in metrics_dict.values()]
    
    df_comp = pd.DataFrame({
        'Model': models,
        'RMSE': rmse_values,
        'R2': r2_values
    })
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='RMSE', data=df_comp)
    plt.title("RMSE Karşılaştırması (Düşük daha iyi)")
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='R2', data=df_comp)
    plt.title("R2 Karşılaştırması (Yüksek daha iyi)")
    
    plt.tight_layout()
    plt.show()

def get_linear_regression_coefficients(model, feature_names):
    if hasattr(model, 'coef_'):
        coefs = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        })
        coefs['Abs_Coef'] = coefs['Coefficient'].abs()
        return coefs.sort_values(by='Abs_Coef', ascending=False)
    else:
        return pd.DataFrame()
