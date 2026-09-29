"""Lab 4: launch a controlled set of runs (one variable changed at a time)."""
import subprocess, sys
from pathlib import Path
script = str(Path(__file__).with_name("train_mlflow.py"))
runs = [
    ["--run_name", "rf_baseline"],
    ["--n_estimators", "50",  "--run_name", "rf_trees_50"],
    ["--n_estimators", "300", "--run_name", "rf_trees_300"],
    ["--max_depth", "5",  "--run_name", "rf_depth_5"],
    ["--max_depth", "20", "--run_name", "rf_depth_20"],
    ["--min_samples_leaf", "3", "--run_name", "rf_leaf_3"],
    ["--model", "gbr", "--n_estimators", "200", "--max_depth", "3", "--run_name", "gbr_n200_d3"],
    ["--model", "ridge", "--alpha", "10", "--run_name", "ridge_alpha10"],
]
for r in runs:
    subprocess.run([sys.executable, script, *r], check=True)
