import unittest
from unittest.mock import patch

import sports_results


def row(series, outcome, line, side):
    return {"clave_interna": repr(((series, outcome, line, None), side))}


def match(home=2, away=1, detail="Final", half_home=1, half_away=0, **extra):
    result = {
        "status": "STATUS_FINAL",
        "completed": True,
        "status_detail": detail,
        "local_score": home,
        "away_score": away,
        "half_local_score": half_home,
        "half_away_score": half_away,
    }
    result.update(extra)
    return result


class ScoreSettlementTests(unittest.TestCase):
    @patch("sports_results.find_match", return_value=match(2, 1))
    def test_home_win_yes(self, _):
        result = sports_results.settle_from_score(row("KXLALIGAGAME", "local", None, "YES"))
        self.assertEqual(result["state"], "WIN")

    @patch("sports_results.find_match", return_value=match(2, 1))
    def test_away_win_no_when_home_does_not_win(self, _):
        result = sports_results.settle_from_score(row("KXLALIGAGAME", "local", None, "NO"))
        self.assertEqual(result["state"], "LOSS")

    @patch("sports_results.find_match", return_value=match(2, 1))
    def test_full_time_over_total(self, _):
        result = sports_results.settle_from_score(row("KXLALIGATOTAL", "over", 2.5, "YES"))
        self.assertEqual(result["state"], "WIN")

    @patch("sports_results.find_match", return_value=match(2, 1))
    def test_team_total(self, _):
        result = sports_results.settle_from_score(row("KXLALIGATEAMTOTAL", "local", 1.5, "YES"))
        self.assertEqual(result["state"], "WIN")

    @patch("sports_results.find_match", return_value=match(2, 1))
    def test_both_teams_to_score(self, _):
        result = sports_results.settle_from_score(row("KXLALIGABTTS", "ambos", None, "YES"))
        self.assertEqual(result["state"], "WIN")

    @patch("sports_results.find_match", return_value=match(2, 1, half_home=1, half_away=0))
    def test_first_half_total_uses_half_time_score_not_full_time(self, _):
        result = sports_results.settle_from_score(row("KXLALIGA1HTOTAL", "over", 0.5, "YES"))
        self.assertEqual(result["state"], "WIN")
        self.assertIn("1.ª parte", result["reason"])
        self.assertIn(": 1-0", result["reason"])

    @patch("sports_results.find_match", return_value=match(2, 1, half_home=1, half_away=0))
    def test_second_half_total_uses_only_second_half_goals(self, _):
        result = sports_results.settle_from_score(row("KXLALIGA2HTOTAL", "over", 1.5, "YES"))
        self.assertEqual(result["state"], "WIN")
        self.assertIn("2.ª parte", result["reason"])
        self.assertIn(": 1-1", result["reason"])

    @patch("sports_results.find_match", return_value=match(2, 1, half_home=1, half_away=0))
    def test_first_half_btts_uses_half_time_score(self, _):
        result = sports_results.settle_from_score(row("KXLALIGA1HBTTS", "ambos", None, "NO"))
        self.assertEqual(result["state"], "WIN")
        self.assertIn("1.ª parte", result["reason"])

    @patch("sports_results.find_match", return_value=match(2, 1, half_home=None, half_away=None))
    def test_half_time_market_without_half_score_defers(self, _):
        result = sports_results.settle_from_score(row("KXLALIGA1HTOTAL", "over", 0.5, "YES"))
        self.assertIsNone(result)

    @patch("sports_results.find_match", return_value=match(0, 0, half_home=0, half_away=0))
    def test_first_team_to_score_no_when_match_finishes_scoreless(self, _):
        result = sports_results.settle_from_score(row("KXLALIGAFTTS", "local", None, "NO"))
        self.assertEqual(result["state"], "WIN")

    @patch("sports_results.find_match", return_value=match(0, 0, half_home=0, half_away=0))
    def test_legacy_firstgoal_series_no_when_match_finishes_scoreless(self, _):
        result = sports_results.settle_from_score(row("KXLALIGAFIRSTGOAL", "local", None, "NO"))
        self.assertEqual(result["state"], "WIN")

    @patch("sports_results._espn_summary", return_value={
        "scoringPlays": [{"team": {"displayName": "Celta"}}]
    })
    @patch("sports_results.find_match", return_value=match(1, 0, event_id="evt2", league="esp.1"))
    def test_first_team_to_score_uses_ordered_espn_scoring_plays(self, _find, _summary):
        captured = row("KXLALIGAFTTS", "visita", None, "YES")
        captured.update({"local": "Elche", "visita": "Celta"})
        result = sports_results.settle_from_score(captured)
        self.assertEqual(result["state"], "WIN")

    @patch("sports_results.find_match", return_value=match(2, 1))
    def test_unsupported_corners_fall_back_when_summary_unavailable(self, _):
        result = sports_results.settle_from_score(row("KXLALIGACORNERS", "over", 8.5, "YES"))
        self.assertIsNone(result)

    @patch("sports_results.find_match", return_value=match(0, 0, half_home=0, half_away=0))
    def test_spread_from_explicit_saved_description(self, _):
        captured = row("KXLALIGA1HSPREAD", "local", 1.0, "NO")
        captured.update({"jugada": "Celta no pierde por más de 1 gol (1.ª parte)", "local": "Elche", "visita": "Celta"})
        result = sports_results.settle_from_score(captured)
        self.assertEqual(result["state"], "LOSS")
        self.assertIn("1.ª parte", result["reason"])

    @patch("sports_results._espn_summary", return_value={
        "boxscore": {"teams": [
            {"team": {"displayName": "Elche"}, "statistics": [{"name": "wonCorners", "displayValue": "5"}]},
            {"team": {"displayName": "Celta"}, "statistics": [{"name": "wonCorners", "displayValue": "2"}]},
        ]}
    })
    @patch("sports_results.find_match", return_value=match(0, 0, half_home=0, half_away=0, event_id="evt1", league="esp.1"))
    def test_total_corners_from_matched_espn_boxscore(self, _find, _summary):
        captured = row("KXLALIGACORNERS", "over", 7.0, "YES")
        captured.update({"local": "Elche", "visita": "Celta"})
        result = sports_results.settle_from_score(captured)
        self.assertEqual(result["state"], "WIN")
        self.assertIn("5-2", result["reason"])

    @patch("sports_results.find_match", return_value=match(2, 1, "Final/Extra Time"))
    def test_extra_time_defers_to_kalshi(self, _):
        result = sports_results.settle_from_score(row("KXUCLGAME", "local", None, "YES"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
