"""Measure the target-reconstruction upper bound from its sensory score components."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import RAW, SENSORY, TARGET, SEED
from preprocess import TEST_SIZE, clean

REPORT_PATH = ROOT / "reports" / "lab13_sensory_benchmark_report.json"


def calculate_benchmark() -> dict:
    frame = clean(pd.read_csv(RAW), verbose=False)
    target = frame[TARGET].to_numpy(dtype=float)
    target_bins = pd.qcut(target, 5, labels=False, duplicates="drop")
    train_indices, test_indices = train_test_split(
        np.arange(len(frame)),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=target_bins,
    )
    omitted_component = "Clean.Cup"
    benchmark_features = [name for name in SENSORY if name != omitted_component]
    features = frame[benchmark_features].to_numpy(dtype=float)
    model = LinearRegression().fit(features[train_indices], target[train_indices])
    actual = target[test_indices]
    predictions = model.predict(features[test_indices])
    if not np.isfinite(actual).all() or not np.isfinite(predictions).all():
        raise ValueError("Sensory benchmark inputs and predictions must be finite.")

    return {
        "benchmark": "illustrative sensory score ablation",
        "target": TARGET,
        "method": "LinearRegression trained on the training split",
        "included_components": benchmark_features,
        "omitted_component": omitted_component,
        "holdout_rows": int(len(test_indices)),
        "holdout_metrics": {
            "mae": float(mean_absolute_error(actual, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
            "r2": float(r2_score(actual, predictions)),
            "max_absolute_error": float(np.max(np.abs(actual - predictions))),
        },
        "target_leakage": True,
        "production_eligible": False,
        "warning": (
            "The included sensory scores are components used to calculate the target. "
            "This intentionally partial reconstruction is an illustrative target-leakage "
            "ablation, not a predictive model or a valid production performance score. "
            "Its R2 is not comparable to the all-component reconstruction or the production model."
        ),
    }


def main() -> None:
    report = calculate_benchmark()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[DIAGNOSTIC ONLY - TARGET LEAKAGE] Holdout R2="
        f"{report['holdout_metrics']['r2']:.6f}; "
        f"report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
