import os
from sklearn.model_selection import train_test_split
from utils import featureSelection
from utils import results_regression
from business import predict_regression
from utils import data_loader
from utils import config_experiment

# CONFIG
print("Setting up Energy Prediction Experiment")
results_path = config_experiment.results_base_path + "results_energy.csv"
dataset_path = config_experiment.dataset_path

test_size = config_experiment.test_size
random_state = config_experiment.random_state
model_list = config_experiment.available_models

use_shap = False
use_hyperparams = False
use_autospearman = True
sampling_mode = "None"

# LOAD & PREPROCESS DATA
print("Loading dataset and preprocessing...")
X, y = data_loader.load_energy_dataset(dataset_path)

if use_autospearman:
    print("Applying AutoSpearman for feature selection...")
    X = featureSelection.autoSpearman(X)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# RUN MODEL(S)
dataset_name = os.path.basename(dataset_path)
configuration = [dataset_name, False, use_autospearman, use_hyperparams, str(X.shape[1])]

for model in model_list:
    print(f"Training model: {model}")
    y_test_eval, y_pred, model_obj, model_name = predict_regression.predict(
        model, use_hyperparams, X_train, X_test, y_train, y_test, random_state, use_shap
    )
    results_regression.display(y_test_eval, y_pred, model_name, results_path, configuration, sampling_mode)
    print(f"Results saved for model: {model} → {results_path}")
