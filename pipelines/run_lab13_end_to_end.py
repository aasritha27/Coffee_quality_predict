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

    run_log["completed_at"] = datetime.now(timezone.utc).isoformat()
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    print(f"\n[{run_log['status']}] Lab 13 run log: {RUN_LOG}")
    return 0 if run_log["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
