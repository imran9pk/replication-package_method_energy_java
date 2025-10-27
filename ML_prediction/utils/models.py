from enum import Enum
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
                             

class ModelType(Enum):
    RandomForestRegressor = 'RandomForestRegressor'
    ADABoostRegressor = 'ADABoostRegressor'
    MLPRegressor = 'MLPRegressor'


def get_model(model_type:ModelType):
    if model_type == ModelType.RandomForestRegressor:
        model = RandomForestRegressor(random_state=42)
    elif model_type == ModelType.ADABoostRegressor:
        model = AdaBoostRegressor(random_state=42)
    elif model_type == ModelType.MLPRegressor:
        model = MLPRegressor(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model
    
    