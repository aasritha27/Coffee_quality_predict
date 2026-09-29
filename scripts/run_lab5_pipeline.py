"""Run the coffee data validation, preprocessing, and output checks for Lab 5."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "validate_data.py",
    "preprocess_pipeline.py",
    "validate_outputs.py",
    "validate_reproducibility.py",
)


def main() -> int:
    for name in SCRIPTS:
        script = ROOT / "src" / name
        print(f"\n[INFO] Running {script.relative_to(ROOT)}", flush=True)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if result.returncode:
            print(f"[ERROR] Lab 5 stopped: {name} returned {result.returncode}.")
            return result.returncode
    print("\n[SUCCESS] Lab 5 coffee production pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
