"""Validate the raw CQI coffee dataset before preprocessing."""
import json
import sys

import numpy as np
import pandas as pd
from common import ARTIFACTS, NUMERIC, RAW, TARGET, CATEGORICAL


def validate_data() -> bool:
    errors = []
    if not RAW.is_file():
        errors.append(f"Raw dataset not found: {RAW}")
        frame = None
    else:
        frame = pd.read_csv(RAW)

    report = {"dataset": str(RAW), "validation_status": "FAILED", "errors": errors}
    if frame is not None:
        derived = {"Bag.Weight.kg", "Harvest.Year.num"}
        required = set(NUMERIC + CATEGORICAL + [TARGET, "Bag.Weight", "Harvest.Year"]) - derived
        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        if TARGET in frame.columns:
            target = pd.to_numeric(frame[TARGET], errors="coerce")
            invalid_target = target.isna() | ~np.isfinite(target)
            if invalid_target.any():
                errors.append(f"Target has {int(invalid_target.sum())} missing or non-numeric values.")
            valid_count = int((target > 0).sum())
            if valid_count < 2:
                errors.append("At least two positive target values are required.")
        else:
            valid_count = 0

        report.update({
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "positive_target_rows": valid_count,
            "duplicate_rows": int(frame.duplicated().sum()),
            "missing_required_columns": missing_columns,
        })

    report["validation_status"] = "PASSED" if not errors else "FAILED"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS / "data_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors:
        print("[ERROR] Raw coffee data validation failed:")
        for error in errors:
            print(f" - {error}")
        print(f"Report: {report_path}")
        return False

    print(f"[SUCCESS] Validated {report['rows']} rows and {report['columns']} columns.")
    print(f"[INFO] Report saved to {report_path}")
    return True


if __name__ == "__main__":
    sys.exit(0 if validate_data() else 1)
