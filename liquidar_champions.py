"""Liquidación Champions: resultado oficial de Kalshi y aplazamientos oficiales."""
import os
from datetime import datetime
import requests
from pathlib import Path
from prediction_core import liquidate_kalshi_ledger

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "predicciones_champions.csv"
RESULT = ROOT / "resultados_champions.csv"
COLUMNS = ["prediction_id", "estado", "marcador", "resuelto_en"]


def postponed_match_ids(competition):
    token = os.getenv("CLAVE")
    if not token:
        print("AVISO: CLAVE no está configurada; no se pueden comprobar aplazamientos oficiales.")
        return set()
    now = datetime.now()
    season = now.year if now.month >= 7 else now.year - 1
    try:
        response = requests.get(
            f"https://api.football-data.org/v4/competitions/{competition}/matches",
            headers={"X-Auth-Token": token}, params={"season": season}, timeout=20,
        )
        response.raise_for_status()
        matches = response.json().get("matches", [])
        return {str(m["id"]) for m in matches
                if m.get("id") is not None and str(m.get("status", "")).upper() in {"POSTPONED", "CANCELLED", "CANCELED"}}
    except (requests.RequestException, ValueError, TypeError) as exc:
        print(f"AVISO: no se pudo consultar el estado oficial de {competition}: {exc}")
        return set()


if __name__ == "__main__":
    cancelled = postponed_match_ids("CL")
    liquidate_kalshi_ledger(PRED, RESULT, COLUMNS, side_field="lado", cancelled_match_ids=cancelled)
