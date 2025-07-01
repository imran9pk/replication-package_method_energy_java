# %%
import os
from pathlib import Path
import pandas as pd
import shutil

# %%
def collect_method_csvs_from_async_Logs(source_root: Path, dest_dir: Path):
    
    if not source_root.exists():
        print(f"Source directory {source_root} does not exist.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Collecting method CSVs from {source_root} to {dest_dir}")
    collapsed_files = list(source_root.glob('*.collapsed'))
    for file in collapsed_files:
        shutil.copy(file, dest_dir)

    print(f"Total .collapsed files found and copied: {len(collapsed_files)}")

# %%
def get_class_name_from_file(file_path: Path) -> str:
    # Extract the filename from the path
    filename = os.path.basename(file_path)
    # Extract class name
    if "methods_profile_" not in filename:
        raise ValueError("Invalid filename format.")
    class_part = filename.split("methods_profile_")[1]
    class_name = class_part.split("_")[0]

    return class_name

# Filter lines containing the class name
def clean_filter_lines(file_path: Path, class_name:str)-> list[str]:
    lines = file_path.read_text().splitlines()
    
    # Clean up the lines by replacing unwanted characters
    cleaned_lines = [
        line.replace("$",".").replace("::",".").replace("..",".").replace(".lambda","").replace(".Lambda","") for line in lines
    ]

    # Filter lines that contain the class name
    class_lines = [
        line for line in cleaned_lines
        if class_name in line
    ]

    # Clean lines and remove methods not belonging to the class
    method_lines = []
    for line in class_lines:
        # Skip lines that do not contain a time value
        if " " not in line:
            continue
        stack, time_str = line.rsplit(" ", 1)
        time = time_str.strip()
        methods = stack.split(";")
        # Keep only methods from the target class
        class_methods = [m for m in methods if class_name in m]
        if class_methods:
            method_lines.append(";".join(class_methods) + f" {time}")

    return method_lines

def process_method_times(clean_method_lines, class_name):
    # Aggregate execution time per method
    method_times = {}

    for line in clean_method_lines:
        stack, time_str = line.rsplit(" ", 1)
        time_ns = int(time_str)
        print(f"Time in ns: {time_ns}")
        time_ms = time_ns / 1000000  # Convert to milliseconds
        print(f"Time in ms: {time_ms}")
        methods = stack.split(";")
        for method in methods:
            method_times[method] = method_times.get(method, 0) + time_ms

    #create a df from the method_times dictionary
    cols = ["Method_name", "execution_time(ms)"]
    df = pd.DataFrame(method_times.items(), columns=cols)
    df.insert(0, "task", class_name)  # Insert the task name as the first column  
    return df

# %%
# Read the combined CSV and copy the execution_time values to Methods Dataset
def read_and_update_main_dataset(combined_perf_csv, methods_dataset_csv):
    # Load the source and target CSVs
    perf_target_df = pd.read_csv(methods_dataset_csv)
    perf_source_df = pd.read_csv(combined_perf_csv)


    # Get sets of method names
    source_methods = set(perf_source_df['Method_name'].unique())
    target_methods = set(perf_target_df['Method_name'].unique())

    # Calculate matches
    matched_methods = source_methods & target_methods
    unmatched_methods = source_methods - target_methods

    # Print counts
    print(f"Total methods in source: {len(source_methods)}")
    print(f"Total methods in target: {len(target_methods)}")
    print(f"Matched methods (from source found in target): {len(matched_methods)}")
    print(f"Unmatched methods (from source NOT in target): {len(unmatched_methods)}")

    # Rename source execution_time column to avoid overwrite
    perf_source_df = perf_source_df.rename(columns={"execution_time(ms)": "execution_time_temp"})

    # Merge
    perf_target_df = perf_target_df.merge(
        perf_source_df[["Method_name", "execution_time_temp"]],
        on="Method_name",
        how="left"
    )

    # Add or update the execution_time column
    if 'execution_time(ms)' in perf_target_df.columns:
        perf_target_df['execution_time(ms)'] = perf_target_df['execution_time(ms)'].combine_first(perf_target_df['execution_time_temp'])
    else:
        perf_target_df['execution_time(ms)'] = perf_target_df['execution_time_temp']

    # Remove temporary column
    perf_target_df.drop(columns=['execution_time_temp'], inplace=True)
    perf_target_df.drop_duplicates(subset=['Method_name'], inplace=True)
    perf_target_df.to_csv(methods_dataset_csv, index=False)

# %%
if __name__ == "__main__":
    root_dir = Path.cwd().parent
    outputs_dir = root_dir / "outputs"
    data_dir = root_dir / "data"
    async_profiles_dir = outputs_dir / "outputs" 
    collapsed_csv_dir = outputs_dir / "methods_perf_data"

    combined_perf_csv = outputs_dir / "combined_perf.csv"
    methods_dataset_csv = data_dir / "method_static_metrics_temp.csv"

    # Collect all collapsed files from the async profiles directory
    collect_method_csvs_from_async_Logs(async_profiles_dir, collapsed_csv_dir)

    all_dfs = []
    # iterate on all collapsed files in the directory and process them
    for file_path in collapsed_csv_dir.glob('*.collapsed'):

        class_name = get_class_name_from_file(file_path)
        print(f"Processing file: {file_path} for class: {class_name}")

        clean_method_lines = clean_filter_lines(file_path, class_name)
        print(f"Cleaned method lines for class {class_name}: {len(clean_method_lines)} lines")

        method_times_df = process_method_times(clean_method_lines, class_name)
        print(f"Processed {len(method_times_df)} methods for class {class_name}")

        all_dfs.append(method_times_df)
    
    all_times_dfs = pd.concat(all_dfs, ignore_index=True)
    all_times_dfs.to_csv(combined_perf_csv, index=False)

    read_and_update_main_dataset(combined_perf_csv, methods_dataset_csv)


