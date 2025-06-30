import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from utils.config_experiment import dataset_path
from utils.data_loader import load_energy_dataset

# CONFIG
output_dir = "results/shap/"
os.makedirs(output_dir, exist_ok=True)
shap_plot_file = os.path.join(output_dir, "shap_summary_plot.png")

# Load and train
X, y = load_energy_dataset(dataset_path)
X = X.select_dtypes(include=["number"])  # Drop any non-numeric columns
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


# Compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Save SHAP plot
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(shap_plot_file)
plt.close()

print(f"SHAP summary plot saved to: {shap_plot_file}")
