"""Lab 5 entry point: reuse the project's tested coffee preprocessing."""
import subprocess
import sys
from common import ROOT


if __name__ == "__main__":
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "preprocess.py")],
        cwd=ROOT,
        check=True,
    )
