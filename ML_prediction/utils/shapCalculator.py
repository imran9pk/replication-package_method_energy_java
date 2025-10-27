import shap
import matplotlib.pyplot as plt
import os
import shap
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from pathlib import Path

slow_models = (MLPRegressor, KNeighborsRegressor, SVR, AdaBoostRegressor)

def make_plot_file_name(model_name, configuration, kfold, fold_id):

    kFold = "_kFold" if kfold else ""
    fold_id = f"_fold_{fold_id}" if kfold else ""    
    lowVarianceDrop = "_lowVarianceDrop" if configuration[1] else ""
    dropExecTime = "_dropExecTime" if configuration[2] else ""
    rfecv = "_rfecv" if configuration[3] else ""
    kbest = "_kbest" if configuration[4] else ""
    autospearman = "_autospearman" if configuration[5] else ""
    hyperparameters = "_hyperparameters" if configuration[6] else ""
    features = f"_featureCount_{configuration[7]}"
    
    fileName = f"{model_name}{kFold}{fold_id}{lowVarianceDrop}{dropExecTime}{rfecv}{kbest}{autospearman}{hyperparameters}{features}"

    # Build config directory name (skip fold/model/feature info)
    config_parts = [
        "lowVarianceDrop" if configuration[1] else "",
        "noExecTime" if configuration[2] else "",
        "RFECV" if configuration[3] else "",
        f"KBest_{configuration[7]}" if configuration[4] else "",
        "autoSpearman" if configuration[5] else "",
        "hyperparameters" if configuration[6] else "",
    ]
    config_suffix = "_".join([p for p in config_parts if p]) or "default"
    config_dir = f"{'kFold' if kfold else 'simple'}_{config_suffix}"


    return fileName, config_dir

def get_shap_explainer(model, X_test):
    """Returns an appropriate SHAP explainer for the given model."""
    
    # Linear models
    if isinstance(model, (Lasso, Ridge, LinearRegression)):
        return shap.LinearExplainer(model, X_test, feature_perturbation="interventional")
    
    # Slow, general-purpose fallback models
    elif isinstance(model, (MLPRegressor, KNeighborsRegressor, SVR, AdaBoostRegressor)):
        print(f"Using KernelExplainer (slow) for {type(model).__name__}")
        return shap.KernelExplainer(model.predict, shap.sample(X_test, 50))

    # Tree-based or auto-supported models
    else:
        return shap.Explainer(model, X_test)


def calculateShap(model, X_test, task_type, configuration, kfold, fold_id, model_name):

     # if isinstance(model, slow_models):
     #      print(f"Skipping SHAP for slow model: {model_name}")
     #      return



     # Get the filename and configuration-specific folder
     fileName, config_dir = make_plot_file_name(model_name, configuration, kfold, fold_id)

     # Top-level path: kFold or simple
     sub_dir = "kFold" if kfold else "simple"

     # Full SHAP plots path
     plots_dir = Path.cwd() / "plots" / sub_dir / config_dir / "shap"
     plots_dir.mkdir(parents=True, exist_ok=True)

     # Final file path
     plot_file = plots_dir / f"{fileName}_shap.png"

     print(f"Calculating SHAP values for model: {model_name} USING {task_type} task type AND CLASSIFIER: {model}")
     try:
          explainer = get_shap_explainer(model, X_test)
          shap_values = explainer(X_test)
     except Exception as e:
          print(f"Error calculating SHAP values: {e}")
          return 

     if task_type == 'classification':
          # Handle binary and multi-class
          if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
               # Multi-class: select one class
               shap_to_plot = shap_values[:, :, 0]
          else:
               # Binary classification: single output
               shap_to_plot = shap_values
     else:
          # Regression: single output
          shap_to_plot = shap_values

     # Plot SHAP values and save the figure
     shap.summary_plot(shap_to_plot, feature_names=X_test.columns, max_display=44, plot_type='bar', show=False, plot_size=(12, 6))
     plt.tight_layout()

     plt.savefig(plot_file, bbox_inches='tight')
     plt.close()
     print(f"SHAP summary plot saved as {plot_file}")