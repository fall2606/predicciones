import unittest
import numpy as np

from mlb_engine import mercado_probabilidad, clasificar, encontrar_partido, motivo_evento_sin_partido
from mlb_math import score_probability_matrix


class FirstInningWinnerTests(unittest.TestCase):
    def setUp(self):
        axis = np.arange(0, 26)
        matrix = score_probability_matrix(axis, 4.6, 4.1)
        self.game = {
            "matrix": matrix,
            "axis": axis,
            "home": "Los Angeles Dodgers",
            "away": "Milwaukee Brewers",
            "mu_home": 4.6,
            "mu_away": 4.1,
        }

    def test_tie_and_team_win_are_distinct_first_inning_events(self):
        tie = {
            "ticker": "KXMLBINNINGWIN-26OCT112000LADMIL-1-TIE",
            "series_ticker": "KXMLBINNINGWIN",
            "title": "First inning winner",
        }
        dodgers = {
            "ticker": "KXMLBINNINGWIN-26OCT112000LADMIL-1-LAD",
            "series_ticker": "KXMLBINNINGWIN",
            "title": "First inning winner",
        }
        p_tie, state_tie = mercado_probabilidad(tie, self.game)
        p_dodgers, state_dodgers = mercado_probabilidad(dodgers, self.game)
        self.assertEqual(state_tie, "modelado_primera_entrada")
        self.assertEqual(state_dodgers, "modelado_primera_entrada")
        self.assertTrue(0 < p_tie < 1)
        self.assertTrue(0 < p_dodgers < 1)
        self.assertNotAlmostEqual(1 - p_tie, 1 - p_dodgers, places=3)

    def test_first_inning_market_is_classified_explicitly(self):
        self.assertEqual(clasificar("KXMLBINNINGWIN", {"ticker": "KXMLBINNINGWIN-1-TIE", "title": "First inning winner"}), "Ganador de la 1.ª entrada")

    def test_event_ticker_time_disambiguates_same_teams_on_same_utc_date(self):
        games = [
            {"id": "early", "home": "Los Angeles Dodgers", "away": "Milwaukee Brewers", "date": "2026-10-12T00:00:00Z"},
            {"id": "late", "home": "Los Angeles Dodgers", "away": "Milwaukee Brewers", "date": "2026-10-12T21:00:00Z"},
        ]
        event = {"event_ticker": "KXMLBEXTRAS-26OCT112000LADMIL", "title": "Los Angeles D vs Milwaukee: Extra Innings"}
        match = encontrar_partido(event, games)
        self.assertIsNotNone(match)
        self.assertEqual(match["id"], "early")

    def test_esports_series_is_classified_as_not_mlb(self):
        event = {"event_ticker": "KXMLBBGAME-26OCT110230VAMFL", "series_ticker": "KXMLBBGAME", "title": "Team Vamos vs. Team Flash"}
        self.assertIn("no_es_MLB", motivo_evento_sin_partido(event, []))

    def test_unknown_first_inning_outcome_is_not_given_a_probability(self):
        market = {
            "ticker": "KXMLBINNINGWIN-26OCT112000LADMIL-1-XYZ",
            "series_ticker": "KXMLBINNINGWIN",
            "title": "First inning winner",
        }
        probability, state = mercado_probabilidad(market, self.game)
        self.assertIsNone(probability)
        self.assertIn("desconocido", state)


if __name__ == "__main__":
    unittest.main()
