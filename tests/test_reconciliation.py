import unittest
from unittest.mock import patch

from scripts.auditar_reconciliacion import finished_ids


class ReconciliationStatusTests(unittest.TestCase):
    @patch("scripts.auditar_reconciliacion.requests.get")
    def test_mlb_final_postponed_and_cancelled_games_are_terminal_for_audit(self, get):
        class Response:
            def raise_for_status(self):
                return None
            def json(self):
                return {"dates": [{"games": [
                    {"gamePk": 101, "status": {"abstractGameState": "Final", "detailedState": "Final"}},
                    {"gamePk": 102, "status": {"abstractGameState": "Preview", "detailedState": "Postponed"}},
                    {"gamePk": 103, "status": {"abstractGameState": "Preview", "detailedState": "Cancelled"}},
                    {"gamePk": 104, "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"}},
                ]}]}
        get.return_value = Response()
        self.assertEqual(finished_ids("mlb", ""), {"101", "102", "103"})


if __name__ == "__main__":
    unittest.main()
