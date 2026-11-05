import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    ("Lab 5 validated pipeline", ROOT / "scripts" / "run_lab5_pipeline.py"),
    ("Lab 2 baseline training", ROOT / "src" / "train.py"),
    ("Lab 2 evaluation", ROOT / "src" / "evaluate.py"),
    ("Lab 6 model registry", ROOT / "scripts" / "run_lab6_registry.py"),
)


def main() -> int:
    for name, script in STEPS:
        print(f"\n[INFO] {name}", flush=True)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if result.returncode:
            print(f"[ERROR] Lab 7 stopped at {name}.")
            return result.returncode
    print("\n[SUCCESS] Lab 7 end-to-end coffee pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
