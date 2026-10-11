import csv
import tempfile
import unittest
from pathlib import Path

from scripts.reparar_csv_predicciones import (
    CURRENT_FOOTBALL, LEGACY_FOOTBALL, repair_file,
)


class CSVRepairTests(unittest.TestCase):
    def test_mixed_old_and_new_rows_are_repaired_without_losing_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predicciones_laliga.csv"
            old = {
                "prediction_id": "old", "match_id": "m-old",
                "fecha": "2026-10-20T18:00:00+00:00", "local": "A", "visita": "B",
                "categoria": "Goles totales", "jugada": "Más de 2.5 goles",
                "probabilidad_base": "0.7", "probabilidad_calibrada": "0.7",
                "calibracion": "", "precio": "40%", "disponibilidad": "",
                "ticker": "KXREAL-OLD", "clave_interna": "(('KXLALIGATOTAL', 'over', 2.5, 'over'), 'YES')",
                "capturado_en": "2026-10-10T12:00:00+00:00",
            }
            new = {
                "prediction_id": "new", "match_id": "m-new",
                "fecha": "2026-10-21T18:00:00+00:00", "local": "C", "visita": "D",
                "categoria": "A_TENER_EN_CUENTA · Goles totales", "jugada": "Más de 3.5 goles",
                "probabilidad_base": "0.8", "probabilidad_calibrada": "0.8",
                "calibracion": "", "precio": "50%", "disponibilidad": "",
                "ticker": "KXREAL-NEW", "mercado": "Más de 3.5 goles", "line": "",
                "seleccion": "NO", "bloque": "A_TENER_EN_CUENTA", "edge": "",
                "clave_interna": "(('KXLALIGATOTAL', 'over', 3.5, 'over'), 'NO')",
                "capturado_en": "2026-10-10T13:00:00+00:00",
            }
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LEGACY_FOOTBALL)
                writer.writeheader()
                writer.writerow(old)
                f.write(",".join([])) if False else None
                writer2 = csv.writer(f)
                writer2.writerow([new.get(col, "") for col in CURRENT_FOOTBALL])
            summary = repair_file(path, CURRENT_FOOTBALL, LEGACY_FOOTBALL)
            self.assertTrue(summary["repaired"])
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                header = reader.fieldnames
            self.assertEqual(header, CURRENT_FOOTBALL)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["prediction_id"], "old")
            self.assertEqual(rows[0]["line"], "2.5")
            self.assertEqual(rows[0]["seleccion"], "YES")
            self.assertEqual(rows[0]["bloque"], "SENCILLAS")
            self.assertEqual(rows[0]["capturado_en"], old["capturado_en"])
            self.assertEqual(rows[1]["prediction_id"], "new")
            self.assertEqual(rows[1]["line"], "3.5")
            self.assertEqual(rows[1]["bloque"], "A_TENER_EN_CUENTA")
            self.assertEqual(rows[1]["capturado_en"], new["capturado_en"])
            second = repair_file(path, CURRENT_FOOTBALL, LEGACY_FOOTBALL)
            self.assertFalse(second["repaired"])


if __name__ == "__main__":
    unittest.main()
