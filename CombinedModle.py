# combined_model.py

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def train_xgboost(x_train, y_train, param_grid):
    XGboost_model = XGBRegressor(objective='reg:squarederror')
    grid_search = GridSearchCV(XGboost_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    best_XGboost_model = grid_search.best_estimator_
    
    return best_XGboost_model

def train_random_forest(x_train, y_train):
    RandomForest_Model = RandomForestRegressor()
    RandomForest_Model.fit(x_train, y_train)
    return RandomForest_Model

def train_ridge(x_train, y_train):
    Ridge_Model = Ridge()
    Ridge_Model.fit(x_train, y_train)
    return Ridge_Model

def combine_predictions(pred1, pred2, pred3):
    return (pred1 + pred2 + pred3) / 3

def train_and_evaluate(data, type, x_train, y_train, x_test, y_test):
    # Hyperparameter grid for XGBoost
    param_grid = {
        'n_estimators': [200],
        'learning_rate': [0.05],
        'max_depth': [7],
        'subsample': [0.6],
        'colsample_bytree': [1.0],
        'reg_alpha': [0.1],
        'reg_lambda': [1.5]
    }
    
    # Train models
    best_XGboost_model = train_xgboost(x_train, y_train, param_grid)
    RandomForest_Model = train_random_forest(x_train, y_train)
    Ridge_Model = train_ridge(x_train, y_train)
    
    # Make predictions on test set
    pred_XGboost_y_test = best_XGboost_model.predict(x_test)
    pred_RandomForest_y_test = RandomForest_Model.predict(x_test)
    pred_Ridge_y_test = Ridge_Model.predict(x_test)

    # Make predictions on train set
    pred_XGboost_y_train = best_XGboost_model.predict(x_train)
    pred_RandomForest_y_train = RandomForest_Model.predict(x_train)
    pred_Ridge_y_train = Ridge_Model.predict(x_train)

    # Combine predictions
    Combine_test = combine_predictions(pred_RandomForest_y_test, pred_XGboost_y_test, pred_Ridge_y_test)
    Combine_train = combine_predictions(pred_RandomForest_y_train, pred_XGboost_y_train, pred_Ridge_y_train)

    # Evaluate the model
    r2_test = r2_score(y_test, Combine_test)
    r2_train = r2_score(y_train, Combine_train)

    print(f'Combined R^2 Test: {r2_test}')
    print(f'Combined R^2 Train: {r2_train}')
    
    hybrid_pred = (pred_XGboost_y_test + pred_RandomForest_y_test + pred_Ridge_y_test) / 3

    if type == 'potential':
        data.loc[x_test.index, 'Predicted Potential'] = hybrid_pred
    else:
        data.loc[x_test.index, 'Predicted Wage'] = hybrid_pred

    return Combine_test, Combine_train , data
