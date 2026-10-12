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

    def test_missing_ticker_for_finished_match_is_actionable_not_hidden_as_historical(self):
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        row = {"prediction_id":"old-1","match_id":"m1","fecha":past,"ticker":"—","local":"A","visita":"B","bloque":"SENCILLAS"}
        result = _row_report("LaLiga", row, {}, {"m1":"FINISHED"}, "football")
        self.assertIn("sin ticker", result["motivo"])
        self.assertEqual(result["clasificacion"], "PENDIENTE_ACTIVA")
        self.assertEqual(result["cuenta_alerta"], "SI")

    def test_finished_prediction_without_ticker_is_not_hidden_as_historical(self):
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        row = {"prediction_id":"no-ticker-final","match_id":"m-final","fecha":past,
               "ticker":"—","local":"A","visita":"B"}
        result = _row_report("LaLiga", row, {}, {"m-final":"FINISHED"}, "football")
        self.assertEqual(result["clasificacion"], "PENDIENTE_ACTIVA")
        self.assertEqual(result["cuenta_alerta"], "SI")
        self.assertIn("sin ticker", result["motivo"])

    def test_overdue_prediction_without_ticker_and_fixture_feed_is_alerted(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
        row = {"prediction_id":"no-ticker-no-feed","match_id":"missing-feed-id","fecha":past,
               "ticker":"—","local":"A","visita":"B"}
        result = _row_report("LaLiga", row, {}, {}, "football")
        self.assertEqual(result["clasificacion"], "PENDIENTE_ACTIVA")
        self.assertEqual(result["cuenta_alerta"], "SI")
        self.assertIn("sin ticker", result["motivo"])

    @patch("scripts.auditar_reconciliacion.kalshi_market")
    def test_kalshi_settled_but_result_not_registered_is_alerted(self, market):
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        market.return_value = {"kind":"ok","status":"settled","result":"yes"}
        row = {"prediction_id":"p1","match_id":"m1","fecha":past,"ticker":"KXTEST-1","local":"A","visita":"B","bloque":"SENCILLAS","seleccion":"YES"}
        result = _row_report("LaLiga", row, {}, {"m1":"FINISHED"}, "football")
        self.assertEqual(result["motivo"], "resultado de Kalshi disponible pero no registrado")
        self.assertEqual(result["cuenta_alerta"], "SI")

    @patch("scripts.auditar_reconciliacion.kalshi_market")
    def test_open_market_after_finished_game_is_alerted_even_inside_grace(self, market):
        recent = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        market.return_value = {"kind":"ok","status":"open","result":""}
        row = {"prediction_id":"p2","match_id":"m2","fecha":recent,"ticker":"KXTEST-2"}
        result = _row_report("MLB", row, {}, {"m2":"FINAL"}, "mlb")
        self.assertIn("revisar fuente alternativa", result["motivo"])
        self.assertEqual(result["clasificacion"], "PENDIENTE_ACTIVA")
        self.assertEqual(result["cuenta_alerta"], "SI")

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

    def test_audit_uses_shared_safe_side_recovery_for_legacy_team_corners(self):
        from scripts.auditar_reconciliacion import captured_side
        row = {"ticker":"KXLALIGATCORNERS-26OCT11RBBOSA-RBB6",
               "local":"Real Betis","visita":"Osasuna",
               "jugada":"0–5 corners de Real Betis","seleccion":"",
               "clave_interna":"(('KXLALIGATCORNERS', 'local', 6.0, 'over'), None)"}
        self.assertEqual(captured_side(row), "NO")

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


    def test_seasonal_market_without_fixture_is_archived_not_counted_as_pending(self):
        row = {"prediction_id":"futures-1","match_id":"","fecha":"",
               "ticker":"KXCITYMLBEXPAND-27AUG01SLC-Y"}
        result = _row_report("MLB", row, {}, {}, "mlb")
        self.assertEqual(result["motivo"], "partido no encontrado")
        self.assertEqual(result["clasificacion"], "HISTORICO_NO_PARTIDO")
        self.assertEqual(result["cuenta_alerta"], "NO")


    @patch("scripts.auditar_reconciliacion.kalshi_market")
    def test_missing_official_fixture_after_normal_match_duration_is_alerted(self, market):
        past = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
        market.return_value = {"kind":"ok","status":"open","result":""}
        row = {"prediction_id":"missing-fixture","match_id":"not-in-feed","fecha":past,
               "ticker":"KXTEST-MISSING","seleccion":"YES"}
        result = _row_report("LaLiga", row, {}, {}, "football")
        self.assertEqual(result["motivo"], "partido no encontrado")
        self.assertEqual(result["cuenta_alerta"], "SI")




    @patch("scripts.auditar_reconciliacion.kalshi_market")
    def test_missing_or_closed_kalshi_market_does_not_hide_finished_prediction(self, market):
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        row = {
            "prediction_id": "market-gone",
            "match_id": "m-gone",
            "fecha": past,
            "ticker": "KXTEST-GONE",
            "local": "A",
            "visita": "B",
            "seleccion": "YES",
        }
        for market_state in (
            {"kind": "missing", "status": "not_found", "result": ""},
            {"kind": "ok", "status": "closed", "result": ""},
        ):
            market.return_value = market_state
            result = _row_report("Champions", row, {}, {"m-gone": "FINISHED"}, "football")
            self.assertEqual(result["clasificacion"], "PENDIENTE_ACTIVA")
            self.assertEqual(result["cuenta_alerta"], "SI")
            self.assertIn("falta resultado deportivo verificable", result["motivo"])



    @patch("scripts.auditar_reconciliacion.SESSION.get")
    def test_nfl_scoreboard_matches_event_ticker_by_teams_and_kickoff(self, get):
        kickoff = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
        class Response:
            def raise_for_status(self):
                return None
            def json(self):
                return {"events": [{
                    "id": "espn-123",
                    "name": "Buffalo Bills at Miami Dolphins",
                    "date": kickoff.isoformat(),
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitions": [{"competitors": [
                        {"homeAway": "away", "team": {"displayName": "Buffalo Bills"}},
                        {"homeAway": "home", "team": {"displayName": "Miami Dolphins"}},
                    ]}],
                }]}
        get.return_value = Response()
        row = {
            "prediction_id": "nfl-1",
            "match_id": "KXNFLGAME-TEST",
            "fecha": kickoff.isoformat(),
            "local": "Buffalo Bills",
            "visita": "Miami Dolphins",
        }
        states = finished_matches("nfl", "", [row])
        self.assertEqual(states["KXNFLGAME-TEST"], "FINAL")


if __name__ == "__main__":
    unittest.main()
