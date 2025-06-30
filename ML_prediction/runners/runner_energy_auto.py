import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils import featureSelection
from utils.results_regression import display
from prediction.business.predict_regression import predict
from utils.config_experiment import (
    dataset_path, results_base_path, test_size, random_state, available_models
)
from prediction.utils.data_loader import load_energy_dataset

# CONFIG
results_path = results_base_path + "results_auto.csv"
model_list = available_models
use_shap = False
use_hyperparams = False
use_autospearman = True
sampling_mode = "None"

# LOAD DATA
X, y = load_energy_dataset(dataset_path)

# FEATURE SELECTION
if use_autospearman:
    X = featureSelection.autoSpearman(X)

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# RUN MODEL(S)
dataset_name = os.path.basename(dataset_path)
configuration = [dataset_name, False, use_autospearman, use_hyperparams, str(X.shape[1])]

for model in model_list:
    y_test_eval, y_pred, model_obj, model_name = predict(
        model, use_hyperparams, X_train, X_test, y_train, y_test, random_state, use_shap
    )
    display(y_test_eval, y_pred, model_name, results_path, configuration, sampling_mode)
