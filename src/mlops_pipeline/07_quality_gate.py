import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import OUTPUTS


def run_quality_gate() -> bool:
    metrics_path = OUTPUTS / "baseline_metrics.json"
    errors = []
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        metrics = {}
        errors.append(f"Could not read baseline metrics at {metrics_path}: {exc}")

    for key in ("mae", "rmse", "r2", "dummy_mae"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"Metric {key!r} is missing or not finite.")
    if not errors and metrics["mae"] >= metrics["dummy_mae"]:
        errors.append("Model MAE does not beat the training-mean dummy baseline.")

    report = {
        "status": "PASSED" if not errors else "FAILED",
        "metrics": metrics,
        "release_checks": [
            "MAE, RMSE, R2, and dummy MAE are finite",
            "Model MAE is lower than dummy-baseline MAE",
        ],
        "errors": errors,
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "quality_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for error in errors:
        print(f"[ERROR] {error}")
    print(f"[{report['status']}] Quality-gate report: {report_path}")
    return not errors


if __name__ == "__main__":
    raise SystemExit(0 if run_quality_gate() else 1)
