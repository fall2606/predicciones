import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import mlb_engine


class MLBThreeBlocksTests(unittest.TestCase):
    def test_main_saves_singles_five_high_and_ten_value_as_independent_rows(self):
        start = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        game = {
            "id": "realistic-mlb-game-id",
            "home": "New York Yankees",
            "away": "Boston Red Sox",
            "date": start,
            "status": "Preview",
        }
        families = [
            ("KXMLBGAME", "Yankees moneyline"),
            ("KXMLBTOTAL", "Over 7.5 total runs"),
            ("KXMLBSPREAD", "Run line +1.5"),
            ("KXMLBTEAMTOTAL", "Home team total over 3.5"),
            ("KXMLBBTTS", "Both teams score"),
            ("KXMLB1H", "First five total over 2.5"),
            ("KXMLBEXACT", "Exact score 4-3"),
            ("KXMLBTOTAL", "Over 9.5 runs alternate"),
            ("KXMLBWIN", "Boston Red Sox wins"),
            ("KXMLBRUNS", "Over 8.5 runs"),
            ("KXMLBSPREAD", "Run line -1.5"),
            ("KXMLBTEAMTOTAL", "Away team total over 2.5"),
            ("KXMLBBTTS", "Both teams score alt"),
            ("KXMLB1H", "First three total over 1.5"),
            ("KXMLBEXACT", "Exact score 5-4"),
        ]
        markets = []
        for index, (series, title) in enumerate(families):
            markets.append({
                "ticker": f"{series}-26OCT12NYYBOS-M{index}",
                "series_ticker": series,
                "title": title,
                "yes_sub_title": title,
                "status": "open",
                "yes_ask_dollars": "0.50",
                "no_ask_dollars": "0.50",
            })
        event = {
            "event_ticker": "KXMLBGAME-26OCT12NYYBOS",
            "title": "New York Yankees vs Boston Red Sox",
            "start_time": start,
            "markets": markets,
        }

        def ranked(rows, min_probability=0.65, high_limit=5, value_limit=10):
            ordered = sorted(rows, key=lambda row: row["_p"], reverse=True)
            return ordered[:high_limit], ordered[:value_limit]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predicciones_mlb.csv"
            with patch.object(mlb_engine, "PRED_FILE", path), \
                 patch.object(mlb_engine, "partidos_proximos", return_value=([game], {})), \
                 patch.object(mlb_engine, "series_mlb", return_value=["KXMLBGAME"]), \
                 patch.object(mlb_engine, "eventos_mlb", return_value=[event]), \
                 patch.object(mlb_engine, "mercado_probabilidad", return_value=(0.72, "modelado")), \
                 patch.object(mlb_engine, "rank_market_candidates", side_effect=ranked):
                mlb_engine.main()
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        counts = {}
        for row in rows:
            counts[row["bloque"]] = counts.get(row["bloque"], 0) + 1
            self.assertTrue(row["ticker"])
            self.assertTrue(row["capturado_en"])
            self.assertGreaterEqual(float(row["probabilidad_modelo"]), 0)
            self.assertLessEqual(float(row["probabilidad_modelo"]), 1)
        self.assertEqual(counts.get("ALTA_PROBABILIDAD"), 5)
        self.assertEqual(counts.get("A_TENER_EN_CUENTA"), 10)
        self.assertGreater(counts.get("SENCILLAS", 0), 0)


if __name__ == "__main__":
    unittest.main()
