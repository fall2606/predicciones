import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

import nfl_engine


class NFLModelTests(unittest.TestCase):
    def setUp(self):
        self.stats = {
            "buffalo bills": {"scored": 28.0, "allowed": 18.0, "games": 5},
            "miami dolphins": {"scored": 21.0, "allowed": 25.0, "games": 5},
        }


    def test_team_stats_queries_current_nfl_season_instead_of_invalid_date_range(self):
        payload = {"events": [{
            "competitions": [{
                "status": {"type": {"completed": True}},
                "competitors": [
                    {"homeAway": "home", "score": "27", "team": {"displayName": "Buffalo Bills"}},
                    {"homeAway": "away", "score": "20", "team": {"displayName": "Miami Dolphins"}},
                ],
            }]
        }]}
        with patch.object(nfl_engine, "_json", return_value=payload) as mocked:
            stats = nfl_engine._team_stats()
        self.assertIn("buffalo bills", stats)
        self.assertEqual(stats["buffalo bills"]["scored"], 27.0)
        self.assertEqual(mocked.call_args.args[0], nfl_engine.ESPN)
        self.assertIn("seasontype", mocked.call_args.args[1])


    def test_merge_game_events_combines_market_series_for_same_fixture(self):
        kickoff = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
        events = [
            {"event_ticker": "KXNFL1HSPREAD-EVENT", "title": "BUF Bills vs MIA Dolphins: 1st Half Spread",
             "start_time": kickoff, "markets": [{"ticker": "KXNFL1HSPREAD-M1", "title": "first half"}]},
            {"event_ticker": "KXNFLGAME-EVENT", "title": "BUF Bills vs MIA Dolphins: Game Winner",
             "start_time": kickoff, "markets": [{"ticker": "KXNFLGAME-M2", "title": "winner"}]},
        ]
        merged = nfl_engine.merge_game_events(events)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["title"], "BUF Bills vs MIA Dolphins")
        self.assertEqual({m["ticker"] for m in merged[0]["markets"]},
                         {"KXNFL1HSPREAD-M1", "KXNFLGAME-M2"})


    def test_event_start_falls_back_to_kalshi_market_close_time(self):
        event = {"markets": [{"close_time": "2026-10-13T20:05:00Z", "expected_expiration_time": "2026-10-12T00:15:00Z"}]}
        self.assertEqual(nfl_engine._event_start(event), "2026-10-12T00:15:00Z")


    def test_build_candidates_uses_espn_kickoff_not_kalshi_expiration(self):
        kickoff = datetime.now(timezone.utc) + timedelta(days=2)
        nfl_engine.SCHEDULES[tuple(sorted(("buffalo bills", "miami dolphins")))] = kickoff.isoformat()
        event = {
            "event_ticker": "KXNFLGAME-EVENT",
            "title": "BUF Bills vs MIA Dolphins: Game Winner",
            "markets": [{
                "ticker": "KXNFLGAME-MONEYLINE",
                "title": "BUF Bills to win",
                "yes_sub_title": "BUF Bills to win",
                "status": "open",
                "close_time": (kickoff + timedelta(days=2)).isoformat(),
                "expected_expiration_time": (kickoff + timedelta(hours=3)).isoformat(),
                "yes_ask_dollars": "0.55",
                "no_ask_dollars": "0.47",
            }],
        }
        candidates = nfl_engine.build_candidates([event], self.stats)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["fecha"], kickoff.isoformat())

    def test_reconcile_exclusions_quarantines_old_period_contracts(self):
        import csv
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            pred_path = Path(folder) / "pred.csv"
            excluded_path = Path(folder) / "excluded.csv"
            kickoff = datetime.now(timezone.utc) - timedelta(hours=1)
            row = {
                "prediction_id": "old-period|market|YES",
                "event_ticker": "KXNFL1HSPREAD-EVENT",
                "ticker": "KXNFL2QSPREAD-MARKET",
                "local": "BUF Bills", "visita": "MIA Dolphins",
                "capturado_en": (kickoff - timedelta(minutes=10)).isoformat(),
            }
            with pred_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            with patch.object(nfl_engine, "PRED_FILE", pred_path), patch.object(nfl_engine, "EXCLUDED_FILE", excluded_path):
                excluded = nfl_engine.reconcile_exclusions(self.stats)
            self.assertIn(row["prediction_id"], excluded)
            self.assertTrue(excluded_path.exists())

    def test_projected_score_uses_nfl_team_scoring_and_defense(self):
        projected = nfl_engine.projected_score("BUF Bills", "MIA Dolphins", self.stats)
        self.assertIsNotNone(projected)
        self.assertGreater(projected[0], projected[1])

    def test_insufficient_team_history_does_not_invent_a_forecast(self):
        sparse = {"buffalo bills": {"scored": 28, "allowed": 18, "games": 1},
                  "miami dolphins": {"scored": 21, "allowed": 25, "games": 4}}
        self.assertIsNone(nfl_engine.projected_score("Buffalo Bills", "Miami Dolphins", sparse))

    def test_candidates_preserve_real_ticker_and_yes_no_sides(self):
        kickoff = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
        event = {
            "event_ticker": "KXNFLGAME-26OCT13BUF MIA".replace(" ", ""),
            "title": "BUF Bills at MIA Dolphins: Game Winner",
            "start_time": kickoff,
            "markets": [{
                "ticker": "KXNFLGAME-26OCT13BUFMIA-BUF",
                "title": "BUF Bills to win",
                "yes_sub_title": "BUF Bills to win",
                "status": "open",
                "yes_ask_dollars": "0.55",
                "no_ask_dollars": "0.47",
            }],
        }
        candidates = nfl_engine.build_candidates([event], self.stats)
        self.assertEqual({row["lado"] for row in candidates}, {"YES", "NO"})
        self.assertTrue(all(row["ticker"] == event["markets"][0]["ticker"] for row in candidates))
        self.assertTrue(all(0 <= row["probabilidad_modelo"] <= 1 for row in candidates))
        self.assertTrue(all(row["precio_captura"] is not None for row in candidates))


    def test_period_market_is_skipped_when_only_full_game_model_exists(self):
        p, category = nfl_engine._market_probability(
            {"ticker": "KXNFL1HSPREAD-TEST", "title": "BUF Bills vs MIA Dolphins: 1st Half Spread",
             "yes_sub_title": "BUF Bills -1.5"},
            "Buffalo Bills", "Miami Dolphins", 27.0, 21.0,
        )
        self.assertIsNone(p)
        self.assertEqual(category, "")


    def test_spread_ladder_is_not_misclassified_as_moneyline(self):
        p, category = nfl_engine._market_probability(
            {"ticker": "KXNFLSPREAD-TEST", "title": "DEN Broncos wins by over 10.5 points",
             "yes_sub_title": "DEN Broncos wins by over 10.5 points"},
            "DEN Broncos", "LA Chargers", 22.9, 17.6,
        )
        self.assertLess(p, 0.5)
        self.assertEqual(category, "Hándicap de puntos")

    def test_second_quarter_spread_is_skipped(self):
        p, category = nfl_engine._market_probability(
            {"ticker": "KXNFL2QSPREAD-TEST", "title": "DEN Broncos wins by over 10.5 points",
             "yes_sub_title": "DEN Broncos wins by over 10.5 points"},
            "DEN Broncos", "LA Chargers", 22.9, 17.6,
        )
        self.assertIsNone(p)
        self.assertEqual(category, "")

    def test_total_points_market_uses_line_and_nfl_score_projection(self):
        p, category = nfl_engine._market_probability(
            {"ticker": "KXNFLTOTAL-TEST", "title": "Over 44.5 total points"},
            "Buffalo Bills", "Miami Dolphins", 27.0, 21.0,
        )
        self.assertGreater(p, 0.5)
        self.assertEqual(category, "Total de puntos")


if __name__ == "__main__":
    unittest.main()
