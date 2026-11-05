import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if shutil.which("docker") is None:
        print("[ERROR] Docker was not found on PATH.")
        return 1
    required = (
        ROOT / "deployment" / "model.pkl",
        ROOT / "deployment" / "preprocessor.pkl",
        ROOT / "deployment" / "model_metadata.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        for path in missing:
            print(f"[ERROR] Missing {path.relative_to(ROOT)}. Complete Lab 10 first.")
        return 1
    result = subprocess.run(
        ["docker", "build", "-t", "coffee-quality-mlops:local", str(ROOT)],
        cwd=ROOT,
    )
    if result.returncode:
        return result.returncode
    print("[SUCCESS] Built coffee-quality-mlops:local.")
    print("Run it with: docker run --rm -p 8000:8000 coffee-quality-mlops:local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
