import pandas as pd
import os
from pathlib import Path

def aggregate_all_5fold_results(directory_path):
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory does not exist: {directory_path}")
        return

    files = list(directory.glob("results5Fold*.csv"))
    print(f"Found {len(files)} files starting with 'results5Fold*.csv' in {directory_path}")
    
    if not files:
        print("No files found starting with results5Fold*.csv")
        return

    drop_cols = ['kFold', 'kFoldID', 'dataset_name']
    group_keys = ['lowVarianceDrop', 'dropExecTime', 'rfecv', 'kbest',
                  'autospearman', 'hyperparameters', 'features', 'sampling', 'model_name']
    metric_cols = ['mse', 'rmse', 'mae', 'mape', 'r2']

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue

            df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])

            if any(col not in df_clean.columns for col in group_keys + metric_cols):
                continue

            df_clean['sampling'] = df_clean['sampling'].fillna("None").astype(str)    
            df_clean[group_keys] = df_clean[group_keys].astype(str)
            
            for col in metric_cols:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

            df_clean = df_clean.dropna(subset=metric_cols, how='all')
            if df_clean.empty:
                continue

            summary = df_clean.groupby(group_keys)[metric_cols].agg(['mean', 'std']).reset_index()
            if summary.empty:
                continue

            summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in summary.columns]
            summary_path = file_path.with_name(file_path.stem + "_summary.csv")
            summary.to_csv(summary_path, index=False)

            print(f"{summary_path.name} saved.")

        except Exception as e:
            print(f"{file_path.name} failed: {e}")

########################Aggregate all 5-fold results########################
root = Path.cwd()
fiveFold_dir = root / "results"

aggregate_all_5fold_results(fiveFold_dir)
