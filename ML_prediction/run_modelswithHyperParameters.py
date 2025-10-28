import sys, runpy

sys.path.insert(0, ".")  # PYTHONPATH=.
scripts = [
    "business/runner_kFold_withHyper_autoSpearman.py",
    "business/runner_kFold_withHyper_default.py",
    "business/runner_kFold_withHyper_KBest_30.py",
    "business/runner_kFold_withHyper_lowVarianceDrop.py",

]


for s in scripts:
    print(f">>> Running {s}")
    # executes the file as if run with __main__
    runpy.run_path(s, run_name="__main__")
