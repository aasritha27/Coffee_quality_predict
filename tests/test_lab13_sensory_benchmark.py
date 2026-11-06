import unittest

from pipelines.run_lab13_sensory_benchmark import calculate_benchmark


class SensoryBenchmarkTests(unittest.TestCase):
    def test_partial_reconstruction_meets_requested_range_and_is_marked_as_leakage(self):
        report = calculate_benchmark()
        self.assertGreaterEqual(report["holdout_metrics"]["r2"], 0.90)
        self.assertLess(report["holdout_metrics"]["r2"], 0.99)
        self.assertTrue(report["target_leakage"])
        self.assertFalse(report["production_eligible"])
        self.assertEqual(report["omitted_component"], "Clean.Cup")


if __name__ == "__main__":
    unittest.main()
