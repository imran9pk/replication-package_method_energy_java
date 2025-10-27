import pandas as pd
from pathlib import Path

def combine_5fold_csvs(directory_path, combined_summary_file, combined_5Fold_resuls_file):

    dir_path = Path(directory_path)
    if not dir_path.exists():
        print(f"Directory does not exist: {directory_path}")
        return

    # All CSVs starting with results5Fold
    all_files = list(dir_path.glob("results5Fold*.csv"))

    # Separate summary and non-summary files
    summary_files = [f for f in all_files if f.name.endswith("_summary.csv")]
    base_files = [f for f in all_files if not f.name.endswith("_summary.csv")]

    print(f"Found {len(all_files)} files starting with 'results5Fold*.csv' in {directory_path}")
    print(f"Found {len(summary_files)} summary files and {len(base_files)} base files.")

    # Combine summary files
    if summary_files:
        df_summary = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)
        df_summary.to_csv(combined_summary_file, index=False)
        print(f"Combined summary saved to: {combined_summary_file.name}")
    else:
        print("No _summary.csv files found.")

    # Combine all other result files
    if base_files:
        df_all = pd.concat([pd.read_csv(f) for f in base_files], ignore_index=True)
        df_all.to_csv(combined_5Fold_resuls_file, index=False)
        print(f"Combined raw results saved to: {combined_5Fold_resuls_file.name}")
    else:
        print("No base results5Fold*.csv files found.")


root_dir = Path.cwd()
fiveFold_dir = root_dir / "results"
combined_summary_file = fiveFold_dir / "results5Fold_summary_combined.csv"
combined_5Fold_resuls_file = fiveFold_dir / "results5Fold_all_combined.csv"
combine_5fold_csvs(fiveFold_dir, combined_summary_file, combined_5Fold_resuls_file)

