import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import randint, uniform
from utils import data_loader
from utils import featureSelection, config_experiment, data_loader

# === CONFIG (user-controlled) ===
model_name = "DT"  # Choose: "RF", "DT", "GB", "KNN"
use_rfecv = True
use_autospearman = False
results_path = config_experiment.results_base_path
dataset_path = config_experiment.dataset_path
random_state = config_experiment.random_state

# === Output Setup ===
suffix = "_rfecv" if use_rfecv else "_autospearman" if use_autospearman else "_all"
output_metrics = os.path.join(results_path, f"results_hyper_{model_name}{suffix}.csv")
output_params = os.path.join(results_path, f"best_params_hyper_{model_name}{suffix}.csv")
os.makedirs(results_path, exist_ok=True)

# === Model & Param Grid Setup ===
models_and_params = {
    "RF": (RandomForestRegressor(random_state=random_state), {
        'n_estimators': [int(x) for x in np.linspace(100, 1000, 10)],
        'max_features': ["sqrt", "log2"],
        'max_depth': [None, 5, 10, 20],
        'min_samples_leaf': randint(1, 5)
    }),
    "GB": (GradientBoostingRegressor(random_state=random_state), {
        'n_estimators': [100, 200, 500],
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': [3, 5, 10],
        'min_samples_leaf': randint(1, 5)
    }),
    "DT": (DecisionTreeRegressor(random_state=random_state), {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }),
    "KNN": (KNeighborsRegressor(), {
        'n_neighbors': randint(3, 15),
        'weights': ["uniform", "distance"],
        'leaf_size': randint(20, 40)
    })
}

model, param_dist = models_and_params[model_name]

# === Load and Preprocess Data ===
X, y = data_loader.load_energy_dataset(dataset_path)
if use_rfecv:
    filepath = config_experiment.rfecv_feature_file
    if os.path.exists(filepath):
        selected = pd.read_csv(filepath, header=None).squeeze().tolist()
    else:
        selected = featureSelection.rfecv(X, y)
        pd.Series(selected).to_csv(filepath, index=False, header=False)
    X = X[selected]

if use_autospearman:
    filepath = config_experiment.autospearman_feature_file
    if os.path.exists(filepath):
        selected = pd.read_csv(filepath, header=None).squeeze().tolist()
        X = X[selected]
    else:
        X = featureSelection.autoSpearman(X)
        selected = X.columns
        pd.Series(selected).to_csv(filepath, index=False, header=False)

X = X.select_dtypes(include=["number"])
X[X.columns] = StandardScaler().fit_transform(X)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# === Hyperparameter Tuning ===
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=25, #2500
    cv=5,
    scoring="neg_mean_absolute_error",
    verbose=2,
    random_state=random_state,
    n_jobs=-1 #30
)

search.fit(X_train, y_train)    
best_model = search.best_estimator_

# === Evaluation ===
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# === Save Results ===

dataset_name = os.path.basename(dataset_path)
num_features = X.shape[1]
sampling_mode = "None"  # or "OVER" if used

metrics_row = [
    dataset_name,
    use_rfecv,
    use_autospearman,
    False,  # hyperparameters (always True if you're tuning — or use a flag)
    num_features,
    sampling_mode,
    model_name,
    mae, rmse, r2
]

metrics_header = [
    "dataset_name", "rfecv", "autospearman", "hyperparameters",
    "features", "sampling", "model_name",
    "MAE", "RMSE", "R2"
]

params_row = {"model": model_name, **search.best_params_}



pd.DataFrame([metrics_row], columns=metrics_header).to_csv(output_metrics, mode='a', index=False, header=not os.path.exists(output_metrics))
pd.DataFrame([params_row]).to_csv(output_params, mode='a', index=False, header=not os.path.exists(output_params))

print(f"Completed hyperparameter tuning for {model_name} with{' RFECV' if use_rfecv else ''}{' AutoSpearman' if use_autospearman else ''}.")
