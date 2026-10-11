import unittest
import numpy as np

import mlb_engine


class MLBMarketMatchingTests(unittest.TestCase):
    def test_abbreviated_kalshi_event_matches_full_statsapi_team_names(self):
        event = {
            "event_ticker": "KXMLBGAME-26OCT112000LADMIL",
            "title": "Game 1: Los Angeles D vs Milwaukee",
            "start_time": "2026-10-11T20:00:00Z",
        }
        games = [{
            "id": "game-1",
            "home": "Los Angeles Dodgers",
            "away": "Milwaukee Brewers",
            "date": "2026-10-11T20:00:00Z",
        }]
        found = mlb_engine.encontrar_partido(event, games)
        self.assertIsNotNone(found)
        self.assertEqual(found["id"], "game-1")

    def test_moneyline_ticker_side_returns_finite_model_probability(self):
        game = {
            "home": "Los Angeles Dodgers",
            "away": "Milwaukee Brewers",
            "axis": np.array([0, 1]),
            "matrix": np.array([[0.2, 0.3], [0.1, 0.4]]),
        }
        market = {
            "ticker": "KXMLBGAME-26OCT112000LADMIL-LAD",
            "series_ticker": "KXMLBGAME",
            "title": "Los Angeles D wins",
        }
        probability, reason = mlb_engine.mercado_probabilidad(market, game)
        self.assertIsNone(reason)
        self.assertTrue(np.isfinite(probability))
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)
        # Lower-triangle home win mass (0.1) plus half of tie mass (0.6).
        self.assertAlmostEqual(probability, 0.4)


if __name__ == "__main__":
    unittest.main()
