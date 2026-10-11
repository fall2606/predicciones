import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from scripts.auditar_reconciliacion import _row_report, finished_matches, latest_results


class ReconciliationStatusTests(unittest.TestCase):
    @patch("scripts.auditar_reconciliacion.SESSION.get")
    def test_mlb_final_postponed_and_cancelled_games_are_recognized(self, get):
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
        states = finished_matches("mlb", "", [])
        self.assertEqual(states["101"], "FINAL")
        self.assertEqual(states["102"], "POSTPONED")
        self.assertEqual(states["103"], "CANCELLED")
        self.assertEqual(states["104"], "SCHEDULED")

    def test_missing_ticker_is_preserved_as_historical_not_alert(self):
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        row = {"prediction_id":"old-1","match_id":"m1","fecha":past,"ticker":"—","local":"A","visita":"B","bloque":"SENCILLAS"}
        result = _row_report("LaLiga", row, {}, {"m1":"FINISHED"}, "football")
        self.assertEqual(result["motivo"], "sin ticker (histórico)")
        self.assertEqual(result["clasificacion"], "HISTORICO_SIN_CONTRATO")
        self.assertEqual(result["cuenta_alerta"], "NO")

    @patch("scripts.auditar_reconciliacion.kalshi_market")
    def test_kalshi_settled_but_result_not_registered_is_alerted(self, market):
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        market.return_value = {"kind":"ok","status":"settled","result":"yes"}
        row = {"prediction_id":"p1","match_id":"m1","fecha":past,"ticker":"KXTEST-1","local":"A","visita":"B","bloque":"SENCILLAS","seleccion":"YES"}
        result = _row_report("LaLiga", row, {}, {"m1":"FINISHED"}, "football")
        self.assertEqual(result["motivo"], "resultado de Kalshi disponible pero no registrado")
        self.assertEqual(result["cuenta_alerta"], "SI")

    @patch("scripts.auditar_reconciliacion.kalshi_market")
    def test_open_market_inside_grace_does_not_trigger_alert(self, market):
        recent = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        market.return_value = {"kind":"ok","status":"open","result":""}
        row = {"prediction_id":"p2","match_id":"m2","fecha":recent,"ticker":"KXTEST-2"}
        result = _row_report("MLB", row, {}, {"m2":"FINAL"}, "mlb")
        self.assertIn("dentro del plazo", result["motivo"])
        self.assertEqual(result["cuenta_alerta"], "NO")

    @patch("scripts.auditar_reconciliacion.kalshi_market")
    def test_open_market_after_grace_is_alerted(self, market):
        past = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        market.return_value = {"kind":"ok","status":"open","result":""}
        row = {"prediction_id":"p3","match_id":"m3","fecha":past,"ticker":"KXTEST-3"}
        result = _row_report("MLB", row, {}, {"m3":"FINAL"}, "mlb")
        self.assertIn("plazo vencido", result["motivo"])
        self.assertEqual(result["cuenta_alerta"], "SI")

    def test_unknown_state_is_an_actionable_alert_after_grace(self):
        past = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        row = {"prediction_id":"p4","match_id":"m4","fecha":past,"ticker":"KXTEST-4"}
        result = _row_report("MLB", row, {}, {"m4":"IN_PROGRESS"}, "mlb")
        self.assertEqual(result["motivo"], "estado desconocido")
        self.assertEqual(result["cuenta_alerta"], "SI")

    @patch("scripts.auditar_reconciliacion.kalshi_market")
    def test_settled_market_with_missing_captured_side_is_not_guessed(self, market):
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        market.return_value = {"kind":"ok","status":"finalized","result":"no"}
        row = {"prediction_id":"p-side-missing","match_id":"m5","fecha":past,
               "ticker":"KXTEST-5","seleccion":"","clave_interna":"(('KXTEST', None), None)"}
        result = _row_report("LaLiga", row, {}, {"m5":"FINISHED"}, "football")
        self.assertIn("sin selección YES/NO", result["motivo"])
        self.assertEqual(result["cuenta_alerta"], "SI")

    def test_terminal_result_wins_over_later_legacy_nonterminal_row(self):
        from scripts.auditar_reconciliacion import latest_results
        data = [
            {"prediction_id":"p","estado":"WIN","resuelto_en":"2026-10-01T00:00:00+00:00"},
            {"prediction_id":"p","estado":"NO_EVALUABLE","resuelto_en":"2026-10-02T00:00:00+00:00"},
        ]
        self.assertEqual(latest_results(data)["p"]["estado"], "WIN")



if __name__ == "__main__":
    unittest.main()
