from models.randomForestRegressor import rf_regressor
from models.decisionTreeRegressor import dt_regressor
from models.gradientBoostingRegressor import gb_regressor, hgb_regressor
from models.mlpRegressor import mlp_regressor
from models.kNeighborsRegressor import knn_regressor
from models.linearRegressor import lr_regressor


def predict(model_name, hyperparameters, X_train, X_test, y_train, y_test, random_state, use_shap=False):
    if model_name == "RF":
        y_pred, model = rf_regressor(
            hyperparameters, X_train, X_test, y_train,
            max_depth=None, n_estimators=200, max_features='sqrt',
            min_samples_leaf=1, random_state=random_state
        )
        
    elif model_name == "DT":
        y_pred, model = dt_regressor(
            hyperparameters, X_train, X_test, y_train,
            criterion="squared_error", max_depth=None,
            max_features=None, min_samples_leaf=1, random_state=random_state
        )
    
    elif model_name == "GB":
        y_pred, model = gb_regressor(X_train, X_test, y_train, random_state)
    
    elif model_name == "HGB":
        y_pred, model = hgb_regressor(X_train, X_test, y_train, random_state)

    elif model_name == "MLP":
        y_pred, model = mlp_regressor(X_train, X_test, y_train, random_state)

    elif model_name == "KNN":
        y_pred, model = knn_regressor(X_train, X_test, y_train)
    
    elif model_name == "LR":
        y_pred, model = lr_regressor(X_train, X_test, y_train)

    else:
        raise NotImplementedError(f"Model '{model_name}' not yet supported.")

    if use_shap:
        from utils import shapCalculator
        shapCalculator.calculateShap(model, X_test)

    return y_test, y_pred, model, model_name
