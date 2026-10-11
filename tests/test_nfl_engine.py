import unittest
from nfl_engine import _safe_probability, probability_over, probability_margin, fit_elo, predict_match, market_family, normalize_name

class NFLModelTests(unittest.TestCase):
    def test_probability_bounds_and_invalid_inputs(self):
        for p in (0, 0.5, 1):
            self.assertEqual(_safe_probability(p), p)
        for p in ("NaN", "inf", -0.01, 1.01, None):
            with self.assertRaises(ValueError):
                _safe_probability(p)

    def test_over_probability_is_bounded_and_monotonic(self):
        values = [probability_over(x, 44.5, 13.0) for x in (30, 40, 45, 50, 60)]
        self.assertTrue(all(0 <= p <= 1 for p in values))
        self.assertEqual(values, sorted(values, reverse=True))

    def test_elo_predicts_home_advantage_and_keeps_probabilities_valid(self):
        games = [
            {"date":"2025-09-01T00:00:00Z","home":"Kansas City Chiefs","away":"Denver Broncos","home_score":27,"away_score":17},
            {"date":"2025-09-08T00:00:00Z","home":"Kansas City Chiefs","away":"Buffalo Bills","home_score":24,"away_score":20},
        ]
        ratings = fit_elo(games)
        pred = predict_match("Kansas City Chiefs", "Denver Broncos", ratings, games)
        self.assertGreater(pred["p_home_win"], pred["p_away_win"])
        self.assertAlmostEqual(pred["p_home_win"] + pred["p_away_win"], 1.0)
        self.assertTrue(all(0 <= pred[k] <= 1 for k in ("p_home_win","p_away_win")))

    def test_catalog_family_is_conservative(self):
        self.assertEqual(market_family("KXNFLGAME", "Winner"), "winner")
        self.assertEqual(market_family("KXNFLSPREAD", "Spread"), "spread")
        self.assertEqual(market_family("KXNFLTOTAL", "Total points"), "total")
        self.assertIsNone(market_family("KXNFLPLAYERPASSINGYARDS", "Passing yards"))

    def test_name_normalization(self):
        self.assertEqual(normalize_name("New England Patriots"), "new england patriots")

if __name__ == "__main__":
    unittest.main()
