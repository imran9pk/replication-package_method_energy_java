import sys, runpy

sys.path.insert(0, ".")  # ensures local imports resolve properly

# Sequentially execute all post-modeling scripts
scripts = [
    "business/2_summary5Folds.py",
    "business/4_combineCSVs.py",
    "business/5_pick_top_5_configs.py",
    "business/3_plotsAggregate5Folds.py"
]

print("\n========== Running Post-Modeling Aggregation and Visualization ==========\n")

for s in scripts:
    print(f">>> Running {s}")
    try:
        runpy.run_path(s, run_name="__main__")
    except Exception as e:
        print(f"{s} failed with error: {e}")

print("\nPost-modeling analysis completed. All aggregated results and plots are generated.\n")
