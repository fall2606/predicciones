import unittest
from unittest.mock import patch

import sports_results


def row(series, outcome, line, side):
    return {"clave_interna": repr(((series, outcome, line, None), side))}


def match(home=2, away=1, detail="Final"):
    return {
        "status": "STATUS_FINAL",
        "completed": True,
        "status_detail": detail,
        "local_score": home,
        "away_score": away,
    }


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

    @patch("sports_results.find_match", return_value=match(2, 1))
    def test_unsupported_corners_fall_back(self, _):
        result = sports_results.settle_from_score(row("KXLALIGACORNERS", "over", 8.5, "YES"))
        self.assertIsNone(result)

    @patch("sports_results.find_match", return_value=match(2, 1, "Final/Extra Time"))
    def test_extra_time_defers_to_kalshi(self, _):
        result = sports_results.settle_from_score(row("KXUCLGAME", "local", None, "YES"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
