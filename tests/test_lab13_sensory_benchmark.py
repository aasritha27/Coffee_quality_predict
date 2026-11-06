import unittest

from pipelines.run_lab13_sensory_benchmark import calculate_benchmark


class SensoryBenchmarkTests(unittest.TestCase):
    def test_reconstruction_exceeds_target_but_is_marked_as_leakage(self):
        report = calculate_benchmark()
        self.assertGreater(report["holdout_metrics"]["r2"], 0.8)
        self.assertTrue(report["target_leakage"])
        self.assertFalse(report["production_eligible"])


if __name__ == "__main__":
    unittest.main()
