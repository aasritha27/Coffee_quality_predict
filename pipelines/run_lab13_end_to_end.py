"""Run the coffee-quality MLOps workflow from data checks through API inference."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    ("Lab 7 pipeline", ROOT / "pipelines" / "run_lab7_pipeline.py"),
    ("Lab 8 quality gate", ROOT / "pipelines" / "run_lab8_quality_gate.py"),
    ("Deployment preparation", ROOT / "deployment" / "prepare_deployment.py"),
    ("Lab 12 monitoring", ROOT / "pipelines" / "run_lab12_monitoring.py"),
    (
        "Production API integration tests",
        (
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(ROOT / "tests"),
            "-p",
            "test_lab13_integration.py",
            "-v",
        ),
    ),
    (
        "Sensory reconstruction benchmark (diagnostic only; target leakage)",
        ROOT / "pipelines" / "run_lab13_sensory_benchmark.py",
    ),
)
RUN_LOG = ROOT / "logs" / "lab13_end_to_end_run_log.json"


def main() -> int:
    run_log = {
        "pipeline": "Coffee Quality MLOps Lab 13",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": [],
    }
    for name, command in STEPS:
        args = list(command) if isinstance(command, tuple) else [sys.executable, str(command)]
        print(f"\n[INFO] Running {name}", flush=True)
        result = subprocess.run(args, cwd=ROOT, check=False)
        run_log["stages"].append(
            {"stage": name, "status": "PASSED" if result.returncode == 0 else "FAILED",
             "exit_code": result.returncode}
        )
        if result.returncode:
            run_log["status"] = "FAILED"
            break
    else:
        run_log["status"] = "PASSED"

    monitoring_stage = next(
        (stage for stage in run_log["stages"] if stage["stage"] == "Lab 12 monitoring"),
        None,
    )
    if monitoring_stage and monitoring_stage["status"] == "PASSED":
        monitoring_report = json.loads(
            (ROOT / "reports" / "drift_monitoring_report.json").read_text(encoding="utf-8")
        )
        run_log["monitoring"] = {
            "status": monitoring_report["status"],
            "holdout_metrics": monitoring_report["holdout_metrics"],
        }

    benchmark_stage = next(
        (
            stage
            for stage in run_log["stages"]
            if stage["stage"].startswith("Sensory reconstruction benchmark")
        ),
        None,
    )
    if benchmark_stage and benchmark_stage["status"] == "PASSED":
        benchmark_report = json.loads(
            (ROOT / "reports" / "lab13_sensory_benchmark_report.json").read_text(
                encoding="utf-8"
            )
        )
        run_log["diagnostic_benchmark"] = {
            "target_leakage": benchmark_report["target_leakage"],
            "production_eligible": benchmark_report["production_eligible"],
            "holdout_metrics": benchmark_report["holdout_metrics"],
        }

    run_log["completed_at"] = datetime.now(timezone.utc).isoformat()
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    metric_summary = ""
    if "monitoring" in run_log:
        holdout_r2 = run_log["monitoring"]["holdout_metrics"]["r2"]
        metric_summary = f" | holdout R2={holdout_r2:.3f}"
    if "diagnostic_benchmark" in run_log:
        benchmark_r2 = run_log["diagnostic_benchmark"]["holdout_metrics"]["r2"]
        metric_summary += f" | diagnostic-only R2={benchmark_r2:.3f} (target leakage)"
    print(f"\n[{run_log['status']}{metric_summary}] Lab 13 run log: {RUN_LOG}")
    return 0 if run_log["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
