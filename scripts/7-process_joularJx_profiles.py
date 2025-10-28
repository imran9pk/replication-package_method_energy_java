# %%
from pathlib import Path
import os
import pandas as pd
import shutil

# %%
def collect_method_csvs_from_joularJx_Logs(source_root: Path, dest_dir: Path):
    
    if not source_root.exists():
        print(f"Source directory {source_root} does not exist.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)

    total_dirs = 0
    csv_count = 0

    for timestamped_dir in source_root.iterdir():
        if not timestamped_dir.is_dir():
            continue
        methods_dir = timestamped_dir / 'app' / 'total' / 'methods'
        if methods_dir.exists() and methods_dir.is_dir():
            total_dirs += 1
            csv_files = list(methods_dir.glob('*.csv'))
            for csv_file in csv_files:
                csv_count += 1
                shutil.copy(csv_file, dest_dir)

    print(f"Total timestamped directories found: {total_dirs}")
    print(f"Total CSV files copied: {csv_count}")

#Read and combine all csvs and return as a single dataframe
def read_all_csvs(path: Path):
    dataframe_list = []
    file_count = 0
    for file_path in path.glob("*.csv"):        
        df = pd.read_csv(file_path, header=None, names=["Method_name", "energy(joules)"])
        file_count += 1
        dataframe_list.append(df)
    
    if not dataframe_list:
        print("No CSV files found.")
        return pd.DataFrame(columns=["Method_name", "energy(joules)"]), 0
    
    # Concatenate all dataframes into one
    combined_csvs_df = pd.concat(dataframe_list, ignore_index=True)
    return combined_csvs_df, file_count

def process_all_energy_dfs(combined_df):

    method_exclusions = combined_df["Method_name"].str.startswith(("java", "sun", "jdk", "LoopRunner", "com", "org"))
    excluded_methods_count = method_exclusions.sum()
    print(f"Number of Java/system methods filtered: {excluded_methods_count}")
    
    combined_df = combined_df[~method_exclusions].copy()
    print(f"Total Methods after filter: {len(combined_df)}")

    combined_df["Method_name"] = combined_df["Method_name"].str.replace(r'\$', '.', regex=True)

    grouped_df = combined_df.groupby("Method_name", as_index=False).mean()

    # from the Method names remove "s2subjects."
    grouped_df["Method_name"] = grouped_df["Method_name"].str.replace(r'^s2subjects\.', '', regex=True)

    return grouped_df

# Read the combined CSV and copy the energy values to Methods Dataset
def read_and_update_main_dataset(combined_energy_csv, methods_dataset_csv):
    print(f"Reading combined energy CSV: {combined_energy_csv}")
    # Load the source and target CSVs
    energy_target_df = pd.read_csv(methods_dataset_csv)
    energy_source_df = pd.read_csv(combined_energy_csv)

    # Get sets of method names
    source_methods = set(energy_source_df['Method_name'].unique())
    target_methods = set(energy_target_df['Method_name'].unique())

    # Calculate matches
    matched_methods = source_methods & target_methods
    unmatched_methods = source_methods - target_methods

    # Print counts
    print(f"Total methods in source data: {len(source_methods)}")
    print(f"Total methods in main Dataset: {len(target_methods)}")
    print(f"Matched methods (from source found in main dataset): {len(matched_methods)}")
    print(f"Unmatched methods (from source NOT in main dataset): {len(unmatched_methods)}")
    print(f"Unmatched methods: {unmatched_methods}")


    # Rename source energy column to avoid overwrite
    energy_source_df = energy_source_df.rename(columns={"energy(joules)": "energy_temp"})

    # Merge
    energy_target_df = energy_target_df.merge(
        energy_source_df[["Method_name", "energy_temp"]],
        on="Method_name",
        how="left"
    )

    # Add or update the energy column
    if 'energy(joules)' in energy_target_df.columns:
        energy_target_df['energy(joules)'] = energy_target_df['energy(joules)'].combine_first(energy_target_df['energy_temp'])
    else:
        energy_target_df['energy(joules)'] = energy_target_df['energy_temp']

    # Remove temporary column
    energy_target_df.drop(columns=['energy_temp'], inplace=True)
    unmatched_methods

    energy_target_df.drop_duplicates(subset=['Method_name'], inplace=True)
    energy_target_df.to_csv(methods_dataset_csv, index=False)
    print(f"Updated {methods_dataset_csv} with energy values from {combined_energy_csv}")

# %%
#Main script to read all energy csvs, combine them, and save the combined dataframe
if __name__ == "__main__":
    # Define paths
    root_dir = Path.cwd().parent
    outputs_dir = root_dir / "outputs"
    data_dir = root_dir / "data"
    joular_jx_results = data_dir / "experiment_energy" / "joularjx-result_20iter"
    methods_energy_dir = outputs_dir / "methods_energy_csvs"

    combined_energy_csv  = outputs_dir / "combined_energy.csv"
    energy_output_path = outputs_dir / "methods_energy_csvs"
    methods_dataset_csv = data_dir / "method_static_metrics_All.csv"

    print("Starting energy CSVs processing...")
    collect_method_csvs_from_joularJx_Logs(joular_jx_results, methods_energy_dir)
    combined_df, files_read = read_all_csvs(energy_output_path)
    print(f"Total CSV files read: {files_read}")
    print(f"Total Methods Profiled for Energy: {len(combined_df)}")

    print("Processing combined energy DataFrame...")
    processed_df  = process_all_energy_dfs(combined_df)

    print("Saving processed DataFrame to CSV...")
    processed_df.to_csv(combined_energy_csv, index=False)

    print("Reading and updating main dataset with energy values...")
    read_and_update_main_dataset(combined_energy_csv, methods_dataset_csv)


