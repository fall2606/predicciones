import unittest
from unittest.mock import patch
import numpy as np

import mlb_engine


class MLBMarketMatchingTests(unittest.TestCase):
    def test_extreme_finite_api_run_rate_cannot_create_nan_probability_matrix(self):
        game = {
            "gamePk": 123,
            "gameDate": "2026-10-12T20:00:00Z",
            "status": {"abstractGameState": "Preview"},
            "teams": {
                "home": {"team": {"id": 1, "name": "Home"}},
                "away": {"team": {"id": 2, "name": "Away"}},
            },
        }
        # Finite but absurd API rates make the old 0..25 truncated PMFs underflow to sum=0.
        stats = {
            1: {"name": "Home", "rpg": 1_000_000.0, "games": 100.0},
            2: {"name": "Away", "rpg": 1_000_000.0, "games": 100.0},
        }
        modeled = mlb_engine.modelo_partido(game, stats)
        self.assertIsNotNone(modeled)
        self.assertTrue(np.all(np.isfinite(modeled["matrix"])))
        self.assertAlmostEqual(float(modeled["matrix"].sum()), 1.0, places=10)


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
        self.assertEqual(reason, "modelado")
        self.assertTrue(np.isfinite(probability))
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)
        # Lower-triangle home win mass (0.1) plus half of tie mass (0.6).
        self.assertAlmostEqual(probability, 0.4)

    def test_kalshi_team_aliases_match_abbreviated_tickers(self):
        event = {
            "event_ticker": "KXMLBGAME-26OCT112000LADMIL",
            "title": "Game 1: Los Angeles D vs Milwaukee",
            "start_time": "2026-10-11T20:00:00Z",
        }
        games = [{
            "id": "849809",
            "home": "Los Angeles Dodgers",
            "away": "Milwaukee Brewers",
            "date": "2026-10-11T20:00:00Z",
        }]
        self.assertEqual(mlb_engine.encontrar_partido(event, games)["id"], "849809")

    def test_run_line_explicit_plus_line_is_parsed(self):
        game = {
            "home": "Los Angeles Dodgers",
            "away": "Milwaukee Brewers",
            "axis": np.array([0, 1, 2]),
            "matrix": np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]]),
        }
        market = {
            "ticker": "KXMLBSPREAD-26OCT112000LADMIL-LAD",
            "series_ticker": "KXMLBSPREAD",
            "title": "Run line +1.5",
        }
        probability, reason = mlb_engine.mercado_probabilidad(market, game)
        self.assertEqual(reason, "modelado")
        self.assertTrue(np.isfinite(probability))
        self.assertGreater(probability, 0)
        self.assertLessEqual(probability, 1)

    def test_f5_total_uses_partial_game_distribution(self):
        game = {
            "home": "Los Angeles Dodgers",
            "away": "Milwaukee Brewers",
            "mu_home": 4.4,
            "mu_away": 4.1,
            "axis": np.arange(0, 26),
            "matrix": np.outer(np.ones(26) / 26, np.ones(26) / 26),
        }
        market = {
            "ticker": "KXMLBF5TOTAL-26OCT112000LADMIL-4P5",
            "series_ticker": "KXMLBF5TOTAL",
            "title": "F5 total over 4.5 runs",
        }
        probability, reason = mlb_engine.mercado_probabilidad(market, game)
        self.assertEqual(reason, "modelado")
        self.assertTrue(np.isfinite(probability))
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)

    def test_per_inning_total_is_not_misclassified_as_full_game_total(self):
        game = {
            "home": "Los Angeles Dodgers",
            "away": "Milwaukee Brewers",
            "axis": np.arange(0, 26),
            "matrix": np.outer(np.ones(26) / 26, np.ones(26) / 26),
        }
        market = {
            "ticker": "KXMLBINNINGTOTAL-26OCT112000LADMIL-2-1",
            "series_ticker": "KXMLBINNINGTOTAL",
            "title": "Over 1.5 runs in inning 2",
        }
        self.assertEqual(mlb_engine.clasificar(market["series_ticker"], market), "Entrada individual")
        probability, reason = mlb_engine.mercado_probabilidad(market, game)
        self.assertIsNone(probability)
        self.assertEqual(reason, "sin_modelo_especifico")

    def test_series_level_and_expansion_series_are_not_match_capture_sources(self):
        items = [
            {"ticker": "KXMLBGAME", "title": "MLB Game", "tags": ["sports"]},
            {"ticker": "KXMLBSERIESGAMES", "title": "MLB Series Games", "tags": ["sports"]},
            {"ticker": "KXMLBINNINGTOTAL", "title": "MLB inning total", "tags": ["sports"]},
            {"ticker": "KXCITYMLBEXPAND", "title": "MLB expansion", "tags": ["sports"]},
        ]
        with patch.object(mlb_engine, "sports_series", return_value=items):
            self.assertEqual(mlb_engine.series_mlb(), ["KXMLBGAME"])


if __name__ == "__main__":
    unittest.main()
