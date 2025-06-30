import os
import requests
import yaml
import time
import csv
import re
import unicodedata
import urllib.parse
from pathlib import Path

# Normalizing task names by replacing speacial characters and spaces
def normalize_task_name(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[,'\"“”‘’$#!?%&*]", "", name)
    name = re.sub(r"[()]", "", name)
    name = name.replace(" ", "-").replace("/", "-")
    name = re.sub(r"-{2,}", "-", name)
    return name.strip("-._")

# Load task names from the YAML file hosted on GitHub
def load_task_names_from_github(yaml_url):
    resp = requests.get(yaml_url)
    task_data = yaml.safe_load(resp.text)
    return list(task_data.keys())

# Build CSV from the list of task names
def build_csv_from_tasks(task_names, out_csv):
    row_data = []
    total_available = 0
    total_java_files = 0

    for task_name in task_names:
        norm = normalize_task_name(task_name)
        encoded_path = urllib.parse.quote(f"{norm}/Java")
        api_url = f"{gitHub_api_root}/{encoded_path}"

        try:
            print(f"Checking {task_name} → {norm}")
            resp = requests.get(api_url, headers=HEADERS)
            if resp.status_code != 200:
                row_data.append([task_name, norm, api_url, "", "", False])
                continue

            files = resp.json()
            java_files = [f for f in files if f["name"].endswith(".java")]

            if not java_files:
                row_data.append([task_name, norm, api_url, "", "", False])
                continue

            for jf in java_files:
                java_url = f"{gitHub_raw_base}/{norm}/Java/{jf['name']}"
                row_data.append([task_name, norm, api_url, jf['name'], java_url, True])
                total_java_files += 1

            total_available += 1

        except Exception as e:
            row_data.append([task_name, norm, api_url, "", "", False])
            print(f"{task_name} → {e}")

        time.sleep(0.05)

    # Write to CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "normalized_path", "github_api_url", "java_file_name", "java_raw_url", "available"])
        writer.writerows(row_data)

    # Summary pf the results
    print("\nSUMMARY")
    print(f"Total tasks from YAML   : {len(task_names)}")
    print(f"Tasks with .java files : {total_available}")
    print(f"Total .java files listed: {total_java_files}")
    print(f"CSV saved to            : {out_csv}")


def download_grouped_by_task(input_tasks_urls, code_download_dir):
    count = 0
    skipped = 0
    with open(input_tasks_urls, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["available"].lower() != "true":
                continue

            task = row["normalized_path"]
            java_file = row["java_file_name"]
            url = row["java_raw_url"]

            try:
                task_dir = code_download_dir / task
                os.makedirs(task_dir, exist_ok=True)
            except Exception as e:
                task = "dropped" + "-".join(task.split("-")[-2:])
                task_dir = code_download_dir / task
                os.makedirs(task_dir, exist_ok=True)
            
            output_path = task_dir / java_file

            if os.path.exists(output_path):
                print(f"Already exists: {task}/{java_file}")
                skipped += 1
                continue

            try:
                r = requests.get(url)
                if r.status_code == 200:
                    with open(output_path, "w", encoding="utf-8") as out:
                        out.write(r.text)
                    print(f"Downloaded: {task}/{java_file}")
                    count += 1
                else:
                    print(f"Failed ({r.status_code}): {task}/{java_file}")
            except Exception as e:
                print(f"Error: {task}/{java_file} → {e}")

    print(f"\nDownload Summary")
    print(f"Downloaded: {count}")
    print(f"Skipped existing: {skipped}")

# Main execution starts from here
if __name__ == "__main__":
    
    GITHUB_TOKEN = "" # Removing token for the Repo
    HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

    input_tasks_list_YAML = "https://raw.githubusercontent.com/acmeism/RosettaCodeData/main/Conf/task.yaml"
    gitHub_api_root = "https://api.github.com/repos/acmeism/RosettaCodeData/contents/Task"
    gitHub_raw_base = "https://raw.githubusercontent.com/acmeism/RosettaCodeData/main/Task"
    

    root_dir = Path.cwd().parent
    # Create data folder paths
    data_folder = root_dir / "data"
    OUT_CSV = data_folder / "rosetta_java_task_index.csv"
    code_download_dir = data_folder / "rosetta_code_grouped"

    # Ensure directory exists
    os.makedirs(code_download_dir, exist_ok=True)

    if not os.path.exists(OUT_CSV):
        task_names = load_task_names_from_github(input_tasks_list_YAML)
        build_csv_from_tasks(task_names, OUT_CSV)
    else:
        print(f"CSV file already exists: {OUT_CSV}")
    
    input_tasks_urls = OUT_CSV
    download_grouped_by_task(input_tasks_urls, code_download_dir)