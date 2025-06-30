import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
diagnostics_csv = r"results/diagnostics_5fold.csv"
output_dir = r" results/plots"


# === LOAD DATA ===
df = pd.read_csv(diagnostics_csv)
models = df["model"].unique()
folds = sorted(df["fold"].unique())

# === INTERACTIVE PROMPT ===
print("🧪 Models available:", ", ".join(models))
model_choice = input("Enter model to plot (or 'ALL'): ").strip()

print("📦 Folds available:", ", ".join(map(str, folds)))
fold_choice = input("Enter fold to plot (or 'ALL'): ").strip()

# === FILTER ===
df_filtered = df.copy()
if model_choice.upper() != "ALL":
    df_filtered = df_filtered[df_filtered["model"] == model_choice]
if fold_choice.upper() != "ALL":
    df_filtered = df_filtered[df_filtered["fold"] == int(fold_choice)]

print(f"🔎 Plotting {len(df_filtered)} records")

# === PREDICTED VS ACTUAL ===
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_filtered, x="actual", y="predicted", hue="model", alpha=0.7)
plt.plot([df_filtered["actual"].min(), df_filtered["actual"].max()],
         [df_filtered["actual"].min(), df_filtered["actual"].max()], 'r--')
plt.title("Predicted vs Actual")
plt.xlabel("True Energy")
plt.ylabel("Predicted Energy")
plt.grid(True)
plt.tight_layout()
plt.show()

# === RESIDUAL HISTOGRAM ===
plt.figure(figsize=(6, 4))
sns.histplot(df_filtered["residual"], kde=True, bins=20)
plt.title("Residual Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === PERCENT ERROR HISTOGRAM ===
plt.figure(figsize=(6, 4))
sns.histplot(df_filtered["percent_error"], kde=True, bins=20)
plt.title("Percent Error Distribution")
plt.xlabel("Percent Error")
plt.grid(True)
plt.tight_layout()
plt.show()
