import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "mlops_pipeline" / "07_quality_gate.py"


if __name__ == "__main__":
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT)
    raise SystemExit(result.returncode)
