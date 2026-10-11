"""Settle the exact pre-kickoff Kalshi capture fixture after the game has ended."""
from __future__ import annotations

import csv
import tempfile
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nfl_engine as nfl


FIXTURE = ROOT / "tests" / "fixtures" / "nfl_live_capture_2026-10-11.csv"


def main():
    if not FIXTURE.exists():
        raise FileNotFoundError(f"Falta el snapshot real prepartido: {FIXTURE}")
    rows = nfl.read_csv_tolerant(FIXTURE, expected_columns=nfl.PRED_COLUMNS)
    if not rows:
        raise RuntimeError("El snapshot de predicciones NFL está vacío.")
    kickoffs = [nfl.parse_dt(row.get("hora_inicio")) for row in rows]
    kickoff = max(value for value in kickoffs if value is not None)
    now = datetime.now(timezone.utc)
    if now < kickoff + timedelta(hours=4):
        print(
            f"LIVE_SETTLEMENT_PENDING: partido capturado {kickoff.isoformat()}; "
            f"se puede liquidar tras { (kickoff + timedelta(hours=4)).isoformat() }"
        )
        return

    with tempfile.TemporaryDirectory(prefix="nfl-live-settlement-") as directory:
        directory = Path(directory)
        pred_file = directory / "predicciones_nfl.csv"
        result_file = directory / "resultados_nfl.csv"
        nfl._write_report(pred_file, nfl.PRED_COLUMNS, rows)
        nfl.PRED_FILE, nfl.RESULT_FILE = pred_file, result_file
        print("LIVE_CAPTURE_FIXTURE_BEFORE_SETTLEMENT")
        print(pred_file.read_text(encoding="utf-8"))
        summary = nfl.settle()
        result_text = result_file.read_text(encoding="utf-8")
        print("LIVE_CAPTURE_RESULTS_AFTER_SETTLEMENT")
        print(result_text)
        settled = list(csv.DictReader(result_text.splitlines()))
        ids = {row.get("prediction_id") for row in rows}
        matched = [row for row in settled if row.get("prediction_id") in ids]
        pending = [row for row in matched if row.get("estado") not in nfl.TERMINAL]
        if len(matched) != len(ids) or pending:
            raise AssertionError(
                f"El ciclo del partido real no está completo: predicciones={len(ids)}, "
                f"resultados={len(matched)}, pendientes={[row.get('prediction_id') for row in pending]}, resumen={summary}"
            )
        print(
            f"LIVE_FULL_CYCLE_OK match_id={rows[0].get('match_id')} "
            f"predicciones={len(ids)} resultados_terminales={len(matched)} "
            f"estados={sorted({row.get('estado') for row in matched})}"
        )


if __name__ == "__main__":
    main()
