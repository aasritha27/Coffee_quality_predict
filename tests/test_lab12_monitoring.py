import unittest

import numpy as np

from pipelines.run_lab12_monitoring import compare_samples


class MonitoringComparisonTests(unittest.TestCase):
    def test_similar_distributions_are_not_flagged(self):
        sample = np.linspace(0, 1, 200)
        result = compare_samples(sample, sample.copy())
        self.assertEqual(result["status"], "OK")
        self.assertAlmostEqual(result["ks_statistic"], 0.0)

    def test_separated_distributions_are_flagged(self):
        result = compare_samples(np.zeros(100), np.ones(100))
        self.assertEqual(result["status"], "DRIFT")
        self.assertGreater(result["ks_statistic"], 0.9)

    def test_empty_or_non_finite_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            compare_samples(np.array([]), np.array([1.0]))
        with self.assertRaises(ValueError):
            compare_samples(np.array([np.nan]), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
