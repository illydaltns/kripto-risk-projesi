from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def hyperparameter_tuning(X_train, y_train, n_splits=5):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    grid_search.fit(X_train, y_train)

    print(f"En iyi parametreler: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

