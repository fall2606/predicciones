"""Settle only validated pregame NFL predictions using official Kalshi resolutions."""
from pathlib import Path
import csv
import tempfile

from prediction_core import liquidate_kalshi_ledger
from nfl_engine import PRED_FILE, PRED_COLUMNS, RESULT_FILE, RESULT_COLUMNS, EXCLUDED_FILE


def _read(path):
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    excluded = {str(row.get("prediction_id") or "").strip()
                for row in _read(EXCLUDED_FILE)}
    predictions = _read(PRED_FILE)
    valid = [row for row in predictions if str(row.get("prediction_id") or "").strip() not in excluded]
    previous_results = _read(RESULT_FILE)
    previous_by_id = {str(row.get("prediction_id") or "").strip(): row for row in previous_results}

    with tempfile.TemporaryDirectory(prefix="nfl_settlement_") as folder:
        temp_pred = Path(folder) / "predicciones_nfl_validas.csv"
        temp_res = Path(folder) / "resultados_nfl_validos.csv"
        with temp_pred.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PRED_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(valid)
        with temp_res.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(row for row in previous_results
                             if str(row.get("prediction_id") or "").strip() not in excluded)

        liquidate_kalshi_ledger(temp_pred, temp_res, RESULT_COLUMNS, side_field="lado")
        newly_read = _read(temp_res)
        merged = dict(previous_by_id)
        for row in newly_read:
            pid = str(row.get("prediction_id") or "").strip()
            if pid:
                merged[pid] = row
        with RESULT_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged.values())

    print(f"NFL liquidación: predicciones totales={len(predictions)}, excluidas={len(excluded)}, "
          f"elegibles para liquidar={len(valid)}")


if __name__ == "__main__":
    main()
