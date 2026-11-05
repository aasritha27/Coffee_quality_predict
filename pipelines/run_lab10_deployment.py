import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    ROOT / "scripts" / "run_lab6_registry.py",
    ROOT / "src" / "mlops_pipeline" / "07_quality_gate.py",
    ROOT / "deployment" / "prepare_deployment.py",
)


def main() -> int:
    for script in STEPS:
        print(f"\n[INFO] Running {script.relative_to(ROOT)}", flush=True)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if result.returncode:
            print(f"[ERROR] Lab 10 stopped at {script.relative_to(ROOT)}.")
            return result.returncode
    print("\n[SUCCESS] Coffee champion and deployment artifacts are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
