import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(ROOT / "tests"),
            "-p",
            "test_failure_handling.py",
            "-v",
        ],
        cwd=ROOT,
        check=False,
    )
    raise SystemExit(result.returncode)
