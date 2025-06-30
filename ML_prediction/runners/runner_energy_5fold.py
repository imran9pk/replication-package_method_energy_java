import os
import pandas as pd
from sklearn.model_selection import KFold
from utils import featureSelection, results_regression, config_experiment, data_loader
from business import predict_regression
from utils import regression_diagnostics


# CONFIG
results_path = config_experiment.results_base_path + "results_5fold.csv"
dataset_path = config_experiment.dataset_path
test_size = config_experiment.test_size
random_state = config_experiment.random_state
model_list = config_experiment.available_models


use_shap = False
use_hyperparams = False
use_autospearman = False
rfecv = True
k_folds = 5
sampling_mode = "kfold"

# LOAD & PREPROCESS DATA
print("Loading dataset and applying preprocessing...")
X, y = data_loader.load_energy_dataset(dataset_path)

if rfecv:
    filepath=config_experiment.rfecv_feature_file
    if os.path.exists(filepath):
        print(f"Loading cached features from: {filepath}")
        selected = pd.read_csv(filepath, header=None).squeeze().tolist()
    else:
        print(f"Computing features using RFECV")
        selected = featureSelection.rfecv(X, y)
        pd.Series(selected).to_csv(filepath, index=False, header=False)
        print(f"Saved selected features to: {filepath}")
    X = X[selected]

if use_autospearman:
    filepath=config_experiment.autospearman_feature_file
    if os.path.exists(filepath):
        print(f"Loading cached features from: {filepath}")
        selected = pd.read_csv(filepath, header=None).squeeze().tolist()
    else:
        print(f"Computing features using autospearman")
        X = featureSelection.autoSpearman(X)
        selected = X.columns
        pd.Series(selected).to_csv(filepath, index=False, header=False)
        print(f"Saved selected features to: {filepath}")

# K-FOLD CV
kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    configuration = [os.path.basename(dataset_path), rfecv, use_autospearman, use_hyperparams, str(X.shape[1])]

    for model in model_list:
        print(f"Running K-Fold model: {model}")
        y_test_eval, y_pred, model_obj, model_name = predict_regression.predict(
            model, use_hyperparams, X_train, X_test, y_train, y_test, random_state, use_shap
        )

        regression_diagnostics.log_diagnostics(
        y_test_eval, y_pred, model_name,
        output_csv="results/diagnostics_5fold.csv",
        fold_id=fold
        )
        
        results_regression.display(y_test_eval, y_pred, model_name, results_path, configuration, sampling_mode)
        print(f"Fold results saved for model: {model}")
