import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import reintegrar_capturas_mlb as restore


class RestoreMLBCapturesTests(unittest.TestCase):
    def test_restores_original_capture_times_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "predicciones_mlb.csv"
            recovered = root / "predicciones_mlb_capturas_recuperadas.csv"
            with recovered.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=restore.COLUMNS)
                writer.writeheader()
                for pid, ticker, captured, match_id, kickoff in [
                    ("KXMLBINNINGWIN-26OCT112000LADMIL-1-TIE|NO|SENCILLAS",
                     "KXMLBINNINGWIN-26OCT112000LADMIL-1-TIE", "2026-10-11T06:03:28.092748+00:00",
                     "849809", "2026-10-12T00:00:00Z"),
                    ("KXMLBGAME-26OCT151800TBCLE-TB|YES|SENCILLAS",
                     "KXMLBGAME-26OCT151800TBCLE-TB", "2026-10-11T06:03:28.092748+00:00",
                     "849820", "2026-10-13T00:00:00Z"),
                ]:
                    writer.writerow({
                        "prediction_id": pid, "ticker": ticker, "match_id": match_id,
                        "fecha": kickoff, "capturado_en": captured,
                        "probabilidad_modelo": "0.66", "bloque": "SENCILLAS",
                    })
            with pred.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=restore.COLUMNS)
                writer.writeheader()
            with patch.object(restore, "PRED", pred), patch.object(restore, "RECOVERED", recovered):
                restore.main()
                restore.main()
            with pred.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {r["capturado_en"] for r in rows},
                {"2026-10-11T06:03:28.092748+00:00"},
            )
            self.assertEqual({r["prediction_id"] for r in rows}, restore.REQUIRED_IDS)


if __name__ == "__main__":
    unittest.main()
