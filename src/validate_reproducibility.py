"""Check deterministic RandomForestRegressor training on the processed coffee data."""
import json
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import ARTIFACTS, PROC, SEED


def run_reproducibility_test() -> bool:
    X_train = np.load(PROC / "X_train_final.npy")
    X_test = np.load(PROC / "X_test_final.npy")
    y_train = np.load(PROC / "y_train.npy")
    y_test = np.load(PROC / "y_test.npy")

    params = {"n_estimators": 100, "max_depth": 10, "random_state": SEED, "n_jobs": 1}
    predictions = []
    for _ in range(2):
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test))

    first, second = predictions
    identical = bool(np.array_equal(first, second))
    metrics = {
        "mae": float(mean_absolute_error(y_test, first)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, first))),
        "r2": float(r2_score(y_test, first)),
    }
    report = {
        "test_name": "Coffee regression reproducibility validation",
        "parameters": params,
        "metrics": metrics,
        "predictions_identical": identical,
        "max_absolute_prediction_difference": float(np.max(np.abs(first - second))),
        "status": "PASSED" if identical else "FAILED",
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS / "reproducibility_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
    print(f"Predictions identical: {identical}. Report: {report_path}")
    return identical


if __name__ == "__main__":
    sys.exit(0 if run_reproducibility_test() else 1)
