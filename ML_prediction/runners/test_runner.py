import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils import featureSelection, results_regression, config_experiment, data_loader
from business import predict_regression

# === CONFIG ===
results_path = config_experiment.results_base_path + "results_5fold_test.csv"
dataset_path = config_experiment.dataset_path
test_size = config_experiment.test_size
random_state = config_experiment.random_state
model_list = config_experiment.available_models

use_shap = False
use_hyperparams = False
use_autospearman = True
k_folds = 5
sampling_mode = "kfold"

# === LOAD & PREPROCESS DATA ===
print("🔁 Loading dataset and applying preprocessing...")
X, y = data_loader.load_energy_dataset(dataset_path)

if use_autospearman:
    print("✨ Applying AutoSpearman feature selection...")
    X = featureSelection.autoSpearman(X)

dataset_name = os.path.basename(dataset_path)
kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

# === K-FOLD CV LOOP ===
for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    print(f"\n📦 Fold {fold}/{k_folds} - Train: {len(train_idx)} / Test: {len(test_idx)}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    configuration = [dataset_name, False, use_autospearman, use_hyperparams, str(X.shape[1])]

    for model in model_list:
        print(f"🚀 Training model: {model}")
        y_test_eval, y_pred, model_obj, model_name = predict_regression.predict(
            model, use_hyperparams, X_train, X_test, y_train, y_test, random_state, use_shap
        )
        results_regression.display(y_test_eval, y_pred, model_name, results_path, configuration, sampling_mode)
        print(f"✅ Fold results saved for model: {model}")

        # === SAMPLE PREDICTIONS ===
        print("\n🔍 Sample predictions vs actual:")
        for actual, pred in zip(y_test_eval[:5], y_pred[:5]):
            print(f"  Actual: {actual:.2f}, Predicted: {pred:.2f}")

        # === PLOT: Prediction vs Actual ===
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test_eval, y_pred, alpha=0.6)
        plt.plot([y_test_eval.min(), y_test_eval.max()], [y_test_eval.min(), y_test_eval.max()], 'r--')
        plt.xlabel("True Energy (Joules)")
        plt.ylabel("Predicted Energy")
        plt.title(f"Fold {fold} - {model} - Predicted vs Actual")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # === PLOT: Residuals ===
        residuals = y_test_eval - y_pred
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Fold {fold} - {model} - Residual Distribution")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
