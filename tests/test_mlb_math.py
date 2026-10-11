import unittest
import numpy as np
from mlb_math import pitcher_stats_from_api, score_probability_matrix


class MLBNaNRegressionTests(unittest.TestCase):
    def test_nan_pitcher_era_is_rejected_at_source(self):
        stat = {"era": "NaN", "inningsPitched": "45.0", "strikeOuts": "50", "hits": "40", "baseOnBalls": "12", "earnedRuns": "20"}
        self.assertIsNone(pitcher_stats_from_api(stat, "Bad ERA"))

    def test_finite_pitcher_stats_are_preserved(self):
        stat = {"era": "3.50", "inningsPitched": "45.0", "strikeOuts": "50", "hits": "40", "baseOnBalls": "12", "earnedRuns": "20"}
        parsed = pitcher_stats_from_api(stat, "Valid Pitcher")
        self.assertEqual(parsed["era"], 3.5)
        self.assertEqual(parsed["nombre"], "Valid Pitcher")

    def test_score_matrix_stays_finite_when_mean_is_nan(self):
        axis = np.arange(0, 21)
        matrix = score_probability_matrix(axis, float("nan"), 4.2)
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=12)

    def test_score_matrix_has_valid_distribution_for_realistic_means(self):
        matrix = score_probability_matrix(np.arange(0, 21), 4.8, 3.9)
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
