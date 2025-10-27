from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np

def plot_bias_variance_scatter(df, top_configs, top_n):
    sns.set(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("Set2", top_n)

    ax.scatter(df["r2_std"], df["r2_mean"], color='gray', alpha=0.45, s=50, label="All Configs", zorder=1)
    ax.scatter([0], [1], color="gold", edgecolor="black", s=150, marker="*", label="Ideal (1.0, 0.0)", zorder=4)

    for i, (_, row) in enumerate(top_configs.iterrows()):
        x, y = row["r2_std_jitter"], row["r2_mean"]
        ax.scatter(x, y, color=palette[i], s=180, edgecolor='black', zorder=5)
        ax.text(x, y, str(row["rank"]), color='black', weight='bold', fontsize=10,
                ha='center', va='center', zorder=6)

    ax.axvline(0.1, linestyle='--', color='gray', lw=1)
    ax.set_xlabel("R² Std (Instability)")
    ax.set_ylabel("R² Mean (Performance)")
    ax.set_title("Top 5 Configurations (Bias–Variance Trade-off with Zoomed Inset)", weight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim(-0.01, df["r2_std"].max() + 0.03)
    ax.set_ylim(df["r2_mean"].min() - 0.05, 1.05)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    axins = inset_axes(ax, width="40%", height="45%", loc='upper right')
    axins.scatter(df["r2_std"], df["r2_mean"], color='gray', alpha=0.25, s=30)
    axins.scatter([0], [1], color="gold", edgecolor="black", s=100, marker="*")

    for i, (_, row) in enumerate(top_configs.iterrows()):
        x, y = row["r2_std_jitter"], row["r2_mean"]
        axins.scatter(x, y, color=palette[i], s=120, edgecolor='black')
        axins.text(x, y, str(row["rank"]), color='black', weight='bold', fontsize=9,
                   ha='center', va='center')

    axins.set_xlim(0.08, 0.14)
    axins.set_ylim(0.44, 0.47)
    axins.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=1)

    plt.tight_layout()
    plt.show()


def bias_variance_top5(csv_path, top_n=5):
    df = pd.read_csv(csv_path)
    df = df[df["dropExecTime"] == False].copy()
    df.drop(columns=["dropExecTime"], inplace=True)

    df["bias_variance_score"] = (1 - df["r2_mean"])**2 + (df["r2_std"])**2
    top_configs = df.sort_values(by="bias_variance_score").head(top_n).copy()
    top_configs["rank"] = range(1, top_n + 1)
    top_configs["r2_std_jitter"] = top_configs["r2_std"] + np.linspace(-0.0015, 0.0015, top_n)

    plot_bias_variance_scatter(df, top_configs, top_n)
    return top_configs


# Main execution
root_dir = Path.cwd()
fiveFold_dir = root_dir / "RESULTS_fiveFold" / "results"
top5_output_csv = fiveFold_dir / "top5_bias_variance_configs.csv"
combined_summary_file = fiveFold_dir / "results5Fold_summary_combined.csv"


top5_df = bias_variance_top5(combined_summary_file, top_n=5)
top5_df.to_csv(top5_output_csv, index=False)
print(f"Top 5 configurations saved to {top5_output_csv}")


