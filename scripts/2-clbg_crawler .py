# %%
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re
from pathlib import Path


def crawl_clbg_site(clbg_page_url):
    # Load CLBG page
    response = requests.get(clbg_page_url)
    # Using html5lib
    soup = BeautifulSoup(response.content, 'html5lib')

    # Collect records by looing over <tr> elements
    records = []
    for tr in soup.find_all("tr"):
        
        a_tag = tr.find("a", href=True)
        if not a_tag:
            continue

        href = a_tag['href']
        if not href.startswith("../program/") or not href.endswith(".html"):
            continue

        # Extract clbg_problem and url
        filename = href.split("/")[-1]
        clbg_problem = filename.replace(".html", "").replace("javavm-", "v").replace("-", "_")
        full_url = urljoin(clbg_page_url, href)
        
        #from tr get the last <td> text and convert to integer N
        last_td = tr.find_all("td")[-1].get_text(strip=True).replace(",", "")
        # convert to integer, if possible
        try:
            N = int(last_td)
        except ValueError:
            N = None  # or handle as needed

        records.append({
            "task": clbg_problem,
            "url": full_url,
            "N": N
        })

    # Create DataFrame and remove duplicates
    df = pd.DataFrame(records).drop_duplicates().reset_index(drop=True)

    # Group and compute N_small, N_large, N_medium
    clbg_agg_df = (
        df.groupby(["task", "url"])["N"]
        .agg(N_small='min', N_large='max', N_medium=lambda x: round(x.mean()))
        .reset_index()
    )

    return clbg_agg_df


# %%


def trim_after_class(command_line: str) -> str:
    """Trim command line to stop after the class name that comes after '-cp'."""
    tokens = command_line.strip().split()
    if '-cp' in tokens:
        cp_index = tokens.index('-cp')
        if cp_index + 2 < len(tokens):
            # Return everything up to and including the class name
            return ' '.join(tokens[:cp_index + 3])
    return command_line  # fallback: return full if pattern not found

def process_benchmark_entry(problem_name: str, url: str, output_dir: str) -> str:

    output_dir = Path(output_dir)
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # ---- Extract Source Code and Command Line ----
        pre_tags = soup.find_all("pre")
        source_code = ""
        command_line = ""

        for pre in pre_tags:
            h2 = pre.find_previous_sibling("div")
            if h2 and "source code" in h2.text.lower():
                source_code = pre.get_text()

            if "COMMAND LINE:" in pre.text and not command_line:
                lines = pre.text.splitlines()
                for i, line in enumerate(lines):
                    if "COMMAND LINE:" in line and i + 1 < len(lines):
                        command_line = lines[i + 1].strip()
                        command_line = trim_after_class(command_line)
                        break

        # ---- Save Java Code ----
        if source_code:
            os.makedirs(output_dir, exist_ok=True)
            file_path = f"{output_dir}/{problem_name}.java"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(source_code)
            print(f"Saved: {file_path}")
        else:
            print(f"Java code not found for {problem_name} at {url}")

        return command_line

    except Exception as e:
        print(f"Error processing {problem_name}: {e}")
        return ""


# %%
if __name__ == "__main__":
    root_dir = Path.cwd().parent
    data_dir = root_dir / "data"
    download_dir = data_dir / "clbg-code"
    csv_path = data_dir / "clbg_downloaded_df.csv"
    clbg_page_url = "https://benchmarksgame-team.pages.debian.net/benchmarksgame/measurements/javavm.html"

    clbg_agg_df = crawl_clbg_site(clbg_page_url)
    # Assuming clbg_agg_df has columns: ['task', 'url']
    clbg_agg_df['command_line'] = clbg_agg_df.apply(
        lambda row: process_benchmark_entry(row['task'], row['url'], download_dir),
        axis=1
    )

    #save
    clbg_agg_df.to_csv(csv_path, index=False)


