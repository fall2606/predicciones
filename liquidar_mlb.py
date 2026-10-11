"""Liquidación MLB independiente y recuperable basada en Kalshi."""
from pathlib import Path
from prediction_core import liquidate_kalshi_ledger

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "predicciones_mlb.csv"
RESULT = ROOT / "resultados_mlb.csv"
COLUMNS = ["prediction_id", "estado", "resultado_kalshi", "marcador", "resuelto_en"]

if __name__ == "__main__":
    liquidate_kalshi_ledger(PRED, RESULT, COLUMNS, side_field="lado")
