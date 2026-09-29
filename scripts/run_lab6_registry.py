"""Train a coffee regression candidate, promote the champion, and report it."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("train_registry.py", "automate_lifecycle.py", "generate_registry_report.py")


def main() -> int:
    for name in SCRIPTS:
        script = ROOT / "src" / name
        print(f"\n[INFO] Running {script.relative_to(ROOT)}", flush=True)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if result.returncode:
            print(f"[ERROR] Lab 6 stopped: {name} returned {result.returncode}.")
            return result.returncode
    print("\n[SUCCESS] Lab 6 model registry pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
