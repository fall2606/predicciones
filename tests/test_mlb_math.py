import unittest
import numpy as np
from mlb_math import pitcher_stats_from_api, score_probability_matrix, summarize_score_probabilities


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

    def test_summary_probabilities_are_readable_and_normalized(self):
        axis = np.arange(0, 21)
        matrix = score_probability_matrix(axis, 5.2, 3.7)
        summary = summarize_score_probabilities(matrix, axis)
        self.assertAlmostEqual(summary["home_win"] + summary["away_win"], 1.0, places=12)
        self.assertGreaterEqual(summary["over_8_5"], 0.0)
        self.assertLessEqual(summary["over_8_5"], 1.0)
        self.assertGreaterEqual(summary["tie_score"], 0.0)

    def test_summary_rejects_invalid_probability_matrix(self):
        with self.assertRaises(ValueError):
            summarize_score_probabilities(np.array([[float("nan")]]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
