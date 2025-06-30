import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from utils import config_experiment
from utils import data_loader

# CONFIG
n_folds = 5
top_k_features = 15
output_csv = "results/feature_importance_5fold.csv"
output_plot = "results/feature_importance_plot.png"

# Load dataset
X, y = data_loader.load_energy_dataset(config_experiment.dataset_path)
kf = KFold(n_splits=n_folds, shuffle=True, random_state=config_experiment.random_state)

feature_scores = pd.DataFrame(index=X.columns)

# Train model across folds and capture feature importances
for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    model = RandomForestRegressor(random_state=config_experiment.random_state)
    model.fit(X_train, y_train)
    feature_scores[f"fold_{fold}"] = model.feature_importances_

# Compute mean and sort
feature_scores["mean_importance"] = feature_scores.mean(axis=1)
feature_scores = feature_scores.sort_values(by="mean_importance", ascending=False)
feature_scores.to_csv(output_csv)

# Plot top k
top_features = feature_scores.head(top_k_features).reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x="mean_importance", y="index", data=top_features)
plt.title(f"Top {top_k_features} Most Important Features (RF - 5 Folds)")
plt.xlabel("Average Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(output_plot)
plt.close()

print(f"Feature importances saved to: {output_csv}")
print(f"Plot saved to: {output_plot}")
