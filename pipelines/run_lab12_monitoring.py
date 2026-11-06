"""Compare training and holdout distributions and report model monitoring metrics."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import MODELS, PROC

ALPHA = 0.01
REPORT_PATH = ROOT / "reports" / "drift_monitoring_report.json"


def compare_samples(reference: np.ndarray, current: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=float).reshape(-1)
    current = np.asarray(current, dtype=float).reshape(-1)
    if not reference.size or not current.size:
        raise ValueError("Both reference and current samples must be non-empty.")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Samples must contain only finite values.")

    result = ks_2samp(reference, current, alternative="two-sided", method="auto")
    return {
        "ks_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "status": "DRIFT" if result.pvalue < ALPHA else "OK",
    }


def run_monitoring() -> dict:
    X_train = np.load(PROC / "X_train_final.npy", allow_pickle=False)
    X_test = np.load(PROC / "X_test_final.npy", allow_pickle=False)
    y_train = np.load(PROC / "y_train.npy", allow_pickle=False)
    y_test = np.load(PROC / "y_test.npy", allow_pickle=False)
    metadata = json.loads((PROC / "dataset_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(MODELS / "random_forest_baseline.pkl")

    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and holdout feature arrays must be 2-D with matching columns.")
    if len(metadata["feature_names"]) != X_train.shape[1]:
        raise ValueError("Dataset feature metadata does not match the saved feature arrays.")
    if len(y_train) != len(X_train) or len(y_test) != len(X_test):
        raise ValueError("Feature and target row counts do not match.")

    feature_drift = []
    for column, name in enumerate(metadata["feature_names"]):
        comparison = compare_samples(X_train[:, column], X_test[:, column])
        feature_drift.append({"feature": name, **comparison})

    train_predictions = np.asarray(model.predict(X_train), dtype=float)
    test_predictions = np.asarray(model.predict(X_test), dtype=float)
    prediction_drift = compare_samples(train_predictions, test_predictions)
    metrics = {
        "mae": float(mean_absolute_error(y_test, test_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        "r2": float(r2_score(y_test, test_predictions)),
        "dummy_mae": float(mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Model monitoring produced non-finite evaluation metrics.")

    drifted_features = [row["feature"] for row in feature_drift if row["status"] == "DRIFT"]
    report = {
        "status": "DRIFT_DETECTED" if drifted_features or prediction_drift["status"] == "DRIFT" else "STABLE",
        "method": "two-sample Kolmogorov-Smirnov test",
        "significance_level": ALPHA,
        "reference": "training feature and prediction distributions",
        "current": "held-out test feature and prediction distributions",
        "features_analyzed": len(feature_drift),
        "features_with_drift": len(drifted_features),
        "drifted_features": drifted_features,
        "prediction_distribution": prediction_drift,
        "holdout_metrics": metrics,
        "holdout_beats_dummy": metrics["mae"] < metrics["dummy_mae"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[{report['status']}] Checked {len(feature_drift)} features; "
        f"{len(drifted_features)} show drift. Report: {REPORT_PATH}"
    )
    return report


if __name__ == "__main__":
    try:
        run_monitoring()
    except (OSError, KeyError, ValueError) as exc:
        print(f"[ERROR] Lab 12 monitoring failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
