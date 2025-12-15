from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, **kwargs):
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs)
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(X_train, y_train, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    param_grid = {
        'n_estimators': [100, 200], 
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # We use neg_mean_squared_error because GridSearchCV maximizes score
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_
