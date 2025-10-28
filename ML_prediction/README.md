# In this folder you will find the pipeline we used for the prediction

This directory contains the reproducible pipeline we used to train, evaluate, and analyze models for method-level energy prediction using the generated dataset of static features and dynamic measurements.


## 1. Requirements

- Python 3.7.5
- All dependencies are listed in **`requirements.txt`**

### Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Directory Structure and Description

This package is structured as follows:

- **business/** – contains runnable Python scripts to start and control the experiments.  
- **data/** – contains the datasets used for training and testing.  
  - `method_dataset_noIter.csv`: dataset generated from a single profiling run (1 iteration).  
  - `method_dataset_20iter.csv`: dataset generated from 20 repeated profiling runs (set `PROFILING_ITERATIONS = 20` in `main.py`).  
- **models/** – includes implementations of all regression models (e.g., RF, GB, ADA, SVM, etc.).  
  Hyperparameter values obtained from tuning are already set in these files.  
- **results/** – stores evaluation outputs, metrics, and aggregated summaries.  
- **utils/** – shared utilities for data preparation, feature selection, hyperparameter grids, SHAP analysis, and logging.  
- **visulizations/** – Jupyter notebooks for reproducing figures (RQ1–RQ3).


## 3. Running the Experiments
All scripts assume you are in the ML_prediction root folder.

### Step 1: Run Baseline Models with Default Configurations
This runs models with default hyperparameters and various feature selectors using 5-fold cross-validation.

```bash
PYTHONPATH=. python3 run_models.py
```

### Step 2: Run Hyperparameter Tuning
This script explores hyperparameter spaces for selected models and feature configurations.

```bash
PYTHONPATH=. python3 run_hyperParameterTuning.py
```

Tuned parameters are logged for each fold and model.

### Step 3: Run Final Models with Best Hyperparameters
After tuning, run the best configurations (top 5) using the final runner:

```bash
PYTHONPATH=. python3 run_modelswithHyperParameters.py
```

Best parameters are preloaded into corresponding files in the models/ folder.

## 4. Visualizations and Result Analysis
All visual analysis notebooks are located in ```visulizations/```.

## 5. Output and Results

- Raw results and per-fold metrics: ```results/```
- Aggregated metrics: via ```2_summary5Folds.py```, ```4_combineCSVs.py```
- Top configurations: ```5_pick_top_5_configs.py```
- Result plots: ```plots/``` and ```results_hyper/```
- SHAP and feature importance (if enabled): inside ```results/```

## Notes
- All datasets are already included.
- All model scripts have preloaded tuned parameters.
- Logging is enabled via TeeLogger.py and saved alongside results.