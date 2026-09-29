"""Lab 4: run the identical config twice and compare metrics AND predictions."""
import subprocess, sys, numpy as np
from pathlib import Path
from common import ARTIFACTS
script = str(Path(__file__).with_name("train_mlflow.py"))
preds = []
for i in (1, 2):
    subprocess.run([sys.executable, script, "--run_name", f"repro_check_{i}"], check=True)
    preds.append(np.load(ARTIFACTS / "test_predictions.npy"))
same = np.array_equal(preds[0], preds[1])
print("Predictions identical across runs:", same, "| max abs diff:", float(np.abs(preds[0]-preds[1]).max()))
sys.exit(0 if same else 1)
