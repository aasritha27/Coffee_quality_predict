import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate_data import validate_data


class CoffeeDataFailureTests(unittest.TestCase):
    def test_missing_dataset_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            report_path = temp_path / "missing_report.json"
            valid = validate_data(temp_path / "not-found.csv", report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertFalse(valid)
        self.assertEqual(report["validation_status"], "FAILED")
        self.assertIn("Raw dataset not found", report["errors"][0])

    def test_missing_columns_are_reported_without_touching_project_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = temp_path / "invalid.csv"
            report_path = temp_path / "invalid_report.json"
            pd.DataFrame({"unrelated": [1]}).to_csv(data_path, index=False)
            valid = validate_data(data_path, report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertFalse(valid)
        self.assertEqual(report["validation_status"], "FAILED")
        self.assertTrue(report["missing_required_columns"])


if __name__ == "__main__":
    unittest.main()
