import sys, runpy

sys.path.insert(0, ".")  # PYTHONPATH=.
scripts = [
    "business/runner_kFold_default.py",
    "business/runner_kFold_KBest_10.py",
    "business/runner_kFold_KBest_20.py",
    "business/runner_kFold_KBest_30.py",
    "business/runner_kFold_lowVarianceDrop.py",
    "business/runner_kFold_noExecutionTime.py",
    "business/runner_kFold_RFECV.py",
    "business/runner_kFold_autoSpearman.py"
]


for s in scripts:
    print(f">>> Running {s}")
    # executes the file as if run with __main__
    runpy.run_path(s, run_name="__main__")
