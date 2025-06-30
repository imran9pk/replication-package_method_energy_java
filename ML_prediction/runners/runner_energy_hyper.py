import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from utils.data_loader import load_energy_dataset
from utils.config_experiment import dataset_path, results_base_path, random_state
from utils import featureSelection, config_experiment
from utils import data_loader


# === CONFIG ===
output_metrics = os.path.join(results_base_path, "results_hyper.csv")
output_params = os.path.join(results_base_path, "best_params_hyper.csv")
os.makedirs(results_base_path, exist_ok=True)
use_autospearman = False
rfecv = True


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


# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# === Define hyperparameter grid for RandomForestRegressor ===
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=2000, num=10)], #default 100
    'max_features': ["auto", "sqrt", "log2", None], #default=”sqrt”
    'max_depth': [3, 5, 10, 20, None], #default=None
    'min_samples_leaf': randint(1, 20), #default=1
}

# === Randomized Search ===
model = RandomForestRegressor(random_state=random_state)
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=2500,
    cv=5,
    scoring="neg_mean_absolute_error",
    verbose=2,
    random_state=random_state,
    n_jobs=30
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

# === Evaluate best model ===
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# === Log results ===
metrics_row = ["RF", rfecv, use_autospearman, mae, rmse, r2]
metrics_header = ["model", "rfecv", "autospearman", "MAE", "RMSE", "R2"]
params_row = {"model": "RF", **search.best_params_}

# Append or create new files
suffix = "_rfecv" if rfecv else "_autospearman" if use_autospearman else "_all"
output_metrics = os.path.join(results_base_path, f"results_hyper{suffix}.csv")
output_params = os.path.join(results_base_path, f"best_params_hyper{suffix}.csv")

pd.DataFrame([metrics_row], columns=metrics_header).to_csv(output_metrics, mode='a', index=False, header=not os.path.exists(output_metrics))
pd.DataFrame([params_row]).to_csv(output_params, mode='a', index=False, header=not os.path.exists(output_params))

print("Hyperparameter tuning complete. Results logged.")
