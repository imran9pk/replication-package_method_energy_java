import numpy as np
from scipy.stats import randint

from utils.models import ModelType

random_forest_regressor_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=2000, num=10)], #default 100
    'max_features': ["sqrt", "log2", None], #default=”sqrt”
    'max_depth': [3, 5, 10, 20, None], #default=None
    'min_samples_leaf': randint(1, 20), #default=1
    'min_samples_split': [2, 5, 10], #default=2
    'bootstrap': [True, False] #default=True
}

ada_regressor_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0, 1.5],
    'loss': ['linear', 'square', 'exponential']
}

#based on Karin et al. "Comparison of sampling methods for imbalanced datasets"
mlp_param_grid_simple = {
    'hidden_layer_sizes': [(100,), (100, 100), (32, 64), (32, 64, 129)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.05],
    'learning_rate': ['constant', 'adaptive']
}

def get_hyperparameter_grid(model_type: ModelType):
    grid = {}
    if model_type == ModelType.RandomForestRegressor:
        grid = random_forest_regressor_grid
    elif model_type == ModelType.ADABoostRegressor:
        grid = ada_regressor_grid
    elif model_type == ModelType.MLPRegressor:
        grid = mlp_param_grid_simple
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return grid
    
def get_scoring_metric(model_type: ModelType):
    if model_type in [ModelType.RandomForestRegressor, ModelType.ADABoostRegressor, ModelType.MLPRegressor]:
        return "r2"  # For regression tasks
    else:
        raise ValueError(f"Unsupported model type for scoring: {model_type}")   

