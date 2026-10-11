"""Liquidación recuperable: solo usa el resultado final del contrato Kalshi."""
from pathlib import Path
from prediction_core import liquidate_kalshi_ledger

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "predicciones_laliga.csv"
RESULT = ROOT / "resultados_laliga.csv"
COLUMNS = ["prediction_id", "estado", "marcador", "resuelto_en"]

if __name__ == "__main__":
    liquidate_kalshi_ledger(PRED, RESULT, COLUMNS, side_field="lado")
