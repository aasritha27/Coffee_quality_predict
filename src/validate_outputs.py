"""Validate arrays and metadata produced by the coffee preprocessing pipeline."""
import json
import sys

import numpy as np
from common import ARTIFACTS, PROC


def validate_outputs() -> bool:
    names = ("X_train_final", "X_test_final", "y_train", "y_test")
    errors = []
    arrays = {}

    for name in names:
        path = PROC / f"{name}.npy"
        try:
            arrays[name] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            errors.append(f"Could not load {path}: {exc}")

    metadata_path = PROC / "dataset_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        metadata = {}
        errors.append(f"Could not load {metadata_path}: {exc}")

    if len(arrays) == len(names):
        X_train, X_test = arrays["X_train_final"], arrays["X_test_final"]
        y_train, y_test = arrays["y_train"], arrays["y_test"]
        if X_train.ndim != 2 or X_test.ndim != 2:
            errors.append("Feature matrices must be two-dimensional.")
        if y_train.ndim != 1 or y_test.ndim != 1:
            errors.append("Target arrays must be one-dimensional.")
        if X_train.shape[1] != X_test.shape[1]:
            errors.append("Training and test feature counts differ.")
        if X_train.shape[0] != y_train.shape[0]:
            errors.append("Training feature and target row counts differ.")
        if X_test.shape[0] != y_test.shape[0]:
            errors.append("Test feature and target row counts differ.")
        for name, values in arrays.items():
            if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
                errors.append(f"{name} contains non-numeric or non-finite values.")

        expected_features = metadata.get("n_features_after_encoding")
        if expected_features is not None and X_train.shape[1] != expected_features:
            errors.append("Feature matrix width does not match dataset metadata.")
        for key, values in (("train_rows", y_train), ("test_rows", y_test)):
            expected_rows = metadata.get(key)
            if expected_rows is not None and len(values) != expected_rows:
                errors.append(f"{key} does not match dataset metadata.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {name: list(values.shape) for name, values in arrays.items()},
        "errors": errors,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS / "preprocessing_summary_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors:
        print("[ERROR] Preprocessing output validation failed:")
        for error in errors:
            print(f" - {error}")
        print(f"Report: {report_path}")
        return False

    print("[SUCCESS] Feature/target arrays and metadata are consistent and finite.")
    print(f"[INFO] Report saved to {report_path}")
    return True


if __name__ == "__main__":
    sys.exit(0 if validate_outputs() else 1)
