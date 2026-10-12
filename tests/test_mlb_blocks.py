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
                "yes_ask_dollars": "0.55",
                "no_ask_dollars": "0.55",
            })
        markets.append({
            "ticker": "KXMLBTOTAL-26OCT12NYYBOS-UNOPENED",
            "series_ticker": "KXMLBTOTAL",
            "title": "Over 10.5 total runs",
            "yes_sub_title": "Over 10.5 total runs",
            "status": "unopened",
        })
        event = {
            "event_ticker": "KXMLBGAME-26OCT12NYYBOS",
            "title": "New York Yankees vs Boston Red Sox",
            "start_time": start,
            "markets": markets,
        }

        def ranked(rows, min_probability=0.65, high_limit=5, value_limit=10):
            ordered = sorted(rows, key=lambda row: row["_p"], reverse=True)
            priced = [row for row in ordered if row["_row"].get("precio_captura")]
            return ordered[:high_limit], priced[:value_limit]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predicciones_mlb.csv"
            with patch.object(mlb_engine, "PRED_FILE", path), \
                 patch.object(mlb_engine, "partidos_proximos", return_value=([game], {})), \
                 patch.object(mlb_engine, "series_mlb", return_value=["KXMLBGAME"]), \
                 patch.object(mlb_engine, "eventos_mlb", return_value=[event]), \
                 patch.object(mlb_engine, "mercado_probabilidad", side_effect=lambda market, game: (0.99 if market.get("status") == "unopened" else 0.72, "modelado")), \
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
        # Unpriced markets must not be stored as predictions or recommendations.
        unpriced = [r for r in rows if r["ticker"] == "KXMLBTOTAL-26OCT12NYYBOS-UNOPENED"]
        self.assertFalse(unpriced)


if __name__ == "__main__":
    unittest.main()


class MLBCalibrationFeedbackTests(unittest.TestCase):
    def test_resolved_mlb_losses_adjust_future_probability_but_pending_do_not(self):
        from probability_calibration import calibrate_probability

        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "predicciones_mlb.csv"
            res = Path(tmp) / "resultados_mlb.csv"
            with pred.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["prediction_id", "categoria", "probabilidad_modelo"])
                writer.writeheader()
                writer.writerows([
                    {"prediction_id": f"settled-{i}", "categoria": "Ganador del partido",
                     "probabilidad_modelo": "0.80"}
                    for i in range(30)
                ])
                writer.writerow({"prediction_id": "pending", "categoria": "Ganador del partido",
                                 "probabilidad_modelo": "0.80"})
            with res.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["prediction_id", "estado"])
                writer.writeheader()
                writer.writerows([
                    {"prediction_id": f"settled-{i}", "estado": "LOSS" if i < 24 else "WIN"}
                    for i in range(30)
                ])
                writer.writerow({"prediction_id": "pending", "estado": "PENDIENTE"})

            adjusted, note = calibrate_probability(
                0.80, "Ganador del partido", pred, res
            )

            self.assertLess(adjusted, 0.80)
            self.assertIn("30 casos comparables", note)

    def test_mlb_calibration_waits_for_minimum_30_resolved_comparable_cases(self):
        from probability_calibration import calibrate_probability

        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "predicciones_mlb.csv"
            res = Path(tmp) / "resultados_mlb.csv"
            with pred.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["prediction_id", "categoria", "probabilidad_modelo"])
                writer.writeheader()
                writer.writerows([
                    {"prediction_id": f"settled-{i}", "categoria": "Total de carreras",
                     "probabilidad_modelo": "0.70"}
                    for i in range(29)
                ])
            with res.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["prediction_id", "estado"])
                writer.writeheader()
                writer.writerows([
                    {"prediction_id": f"settled-{i}", "estado": "WIN"}
                    for i in range(29)
                ])

            adjusted, note = calibrate_probability(0.70, "Total de carreras", pred, res)

            self.assertEqual(adjusted, 0.70)
            self.assertIn("29/30", note)
