import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt

def plot_residuals(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Rating')
    plt.ylabel('Predicted Rating')
    plt.title(f'Residual Plot for {model_name}')
    plt.savefig(f'results/{model_name}/residuals.png')
    plt.close()

def save_metrics(y_true, y_pred, model_name):
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = metrics.root_mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    with open(f'results/{model_name}/metrics.txt', 'w') as f:
        f.write(f'MSE: {mse:.4f}\n')
        f.write(f'MAE: {mae:.4f}\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'R^2: {r2:.4f}\n')

def train_knn(x_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40, 50]
    }
    model = KNeighborsRegressor()
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    return search.fit(x_train, y_train)

def train_linear_regression(x_train, y_train):
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'positive': [True, False]
    }
    model = LinearRegression()
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    return search.fit(x_train, y_train)

def train_random_forest(x_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12, 14, 16],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    model = RandomForestRegressor(random_state=1)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=500,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=1
    )
    return search.fit(x_train, y_train)

def train_xgboost(x_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'min_child_weight': [1, 3, 5, 7, 9, 11, 13, 15],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'reg_alpha': [0, 0.1, 0.5, 1, 2, 5],
        'reg_lambda': [0, 0.1, 0.5, 1, 2, 5]
    }
    model = xgb.XGBRegressor(random_state=1)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=500,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=1
    )
    return search.fit(x_train, y_train)

def main():
    df = pd.read_csv('datasets/filtered_dataset.csv')

    features = ['num_courses', 'num_reviews', 'sentiment']
    X = df[features]
    y = df['average_rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    models = {
        'KNN': train_knn,
        'LinearRegression': train_linear_regression,
        'RandomForest': train_random_forest,
        'XGBoost': train_xgboost
    }
    
    for name, train_func in models.items():
        print(f"Training {name}...")
        
        search = train_func(X_train, y_train)
        best_model = search.best_estimator_ if hasattr(search, 'best_estimator_') else search
        
        joblib.dump(best_model, f'models/{name}.pkl')
        
        y_pred = best_model.predict(X_test)
        
        save_metrics(y_test, y_pred, name)
        plot_residuals(y_test, y_pred, name)
        
        print(f"{name} training completed.")

if __name__ == "__main__":
    main() 