import pandas as pd
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
from collections import defaultdict


def generate_plot_grids_dynamic(base_plot_dir, plot_type="line", plot_scale="log", summary_dir=None):
    base_path = Path(base_plot_dir)
    summary_path = Path(summary_dir) if summary_dir else None

    if not base_path.exists() or not summary_path or not summary_path.exists():
        print("Plot or summary directory is invalid.")
        return

    # Load all summary CSVs ending with _summary.csv
    summary_files = {f.stem.replace("results5Fold_", "").replace("_summary", "").lower(): f 
                     for f in summary_path.glob("results5Fold*_summary.csv")}
    
    print(f"Found {len(summary_files)} 5Fold summary files.")
    
    config_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("kFold")]
    print(f"Found {len(config_dirs)} 5Fold configuration plot directories.")

    for config_dir in config_dirs:
        print(f"Processing configuration directory: {config_dir.name}")
        config_key = config_dir.name.replace("kFold_", "").lower()
        config_key = re.sub(r"_hyperparameters$", "", config_key)

        # Special case normalization for KBest dirs
        if config_key.startswith("KBest"):
            config_key = config_key.replace("kbest_", "kbest")
        
        if config_key == "noExecTime" :
            config_key = "noExecutionTime"

        summary_file = summary_files.get(config_key)

        if not summary_file:
            print(f"No matching summary CSV found for {config_dir.name}")
            continue

        # Load summary CSV
        summary_df = pd.read_csv(summary_file)
        model_plots = defaultdict(list)
        print(f"Loading plot files from {config_dir.name} for plot type: {plot_type}")

        for file in config_dir.glob("*.png"):
            parts = file.stem.split("_")
            if len(parts) < 3:
                continue

            model = parts[0]
            if not file.stem.endswith(f"_{plot_type}_{plot_scale}"):
                continue

            try:
                fold_index = parts.index("fold")
                fold_num = int(parts[fold_index + 1])
                model_plots[model].append((fold_num, file))
            except (ValueError, IndexError):
                continue

        if not model_plots:
            print(f"No matching {plot_type} plot files in {config_dir.name}")
            continue

        output_dir = config_dir / "plot_grids"
        output_dir.mkdir(exist_ok=True)

        for model, file_tuples in model_plots.items():
            # Find matching row in summary CSV
            row = summary_df[summary_df['model_name'] == model]
            if row.empty:
                print(f"Model {model} not found in summary CSV {summary_file.name}")
                continue

            summary_stats = {
                "mse_mean": f"{row['mse_mean'].values[0]:.4f}",
                "mse_std": f"{row['mse_std'].values[0]:.4f}",
                "mape_mean": f"{row['mape_mean'].values[0]:.4f}",
                "mape_std": f"{row['mape_std'].values[0]:.4f}",
                "r2_mean": f"{row['r2_mean'].values[0]:.4f}",
                "r2_std": f"{row['r2_std'].values[0]:.4f}"
            }

            file_tuples = sorted(file_tuples, key=lambda x: x[0])
            files = [f for _, f in file_tuples]
            n = len(file_tuples)
            cols = 3
            rows = (n + cols) // cols

            fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axs = axs.flatten()

            for i, (fold_num, img_path) in enumerate(file_tuples):
                img = mpimg.imread(img_path)
                axs[i].imshow(img)
                axs[i].axis('off')
                axs[i].set_title(f"Fold {fold_num}", fontsize=11, fontweight="bold")


            if n < len(axs):
                ax_summary = axs[n]
                ax_summary.set_xticks([])
                ax_summary.set_yticks([])
                ax_summary.axis('off')

                ax_summary.text(
                    0.5, 0.75,
                    "Performance Summary across 5 Folds",
                    ha='center', va='center', fontsize=12, weight='bold'
                )

                lines = [
                    ("MSE mean", f"{summary_stats['mse_mean']} ± {summary_stats['mse_std']}"),
                    ("MAPE mean", f"{summary_stats['mape_mean']} ± {summary_stats['mape_std']}"),
                    ("R² mean", f"{summary_stats['r2_mean']} ± {summary_stats['r2_std']}")
                ]

                for i, (label, value) in enumerate(lines):
                    ypos = 0.60 - i * 0.12
                    ax_summary.text(0.45, ypos, f"{label:>12}  ",
                                    ha='right', va='center', fontsize=11, family='monospace')
                    ax_summary.text(0.47, ypos, value,
                                    ha='left', va='center', fontsize=11, family='monospace')

            for j in range(n + 1, len(axs)):
                axs[j].axis('off')

            plt.tight_layout()
            grid_file = output_dir / f"{model}_fold_grid_{plot_type}_{plot_scale}.png"
            plt.savefig(grid_file)
            plt.close()
            print(f"Saved grid for {model} ({plot_type}): {grid_file.name}")

########################Generate plot grids########################
root = Path.cwd()
fiveFold_dir = root / "results"
plots_dir = root / "plots" / "kFold"

generate_plot_grids_dynamic(
    base_plot_dir= plots_dir,
    plot_type= "box",  # or "scatter"
    plot_scale = "log", # log or exp
    summary_dir= fiveFold_dir
)
