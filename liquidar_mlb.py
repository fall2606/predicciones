"""Liquidación MLB: resultado oficial de Kalshi y partidos aplazados/cancelados."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from prediction_core import liquidate_kalshi_ledger

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "predicciones_mlb.csv"
RESULT = ROOT / "resultados_mlb.csv"
COLUMNS = ["prediction_id", "estado", "resultado_kalshi", "marcador", "resuelto_en"]


def postponed_match_ids():
    now = datetime.now(timezone.utc).date()
    try:
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "startDate": (now - timedelta(days=14)).isoformat(),
                    "endDate": (now + timedelta(days=30)).isoformat(), "gameTypes": "R,F,D,L,W"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        games = [game for day in payload.get("dates", []) for game in day.get("games", [])]
    except Exception as exc:
        print(f"AVISO: no se pudo consultar el calendario MLB para aplazamientos: {exc}")
        return set()
    cancelled = set()
    for game in games:
        status = game.get("status", {})
        detailed = str(status.get("detailedState", "")).casefold()
        if game.get("gamePk") is not None and detailed in {"postponed", "cancelled", "canceled"}:
            cancelled.add(str(game["gamePk"]))
    return cancelled


if __name__ == "__main__":
    liquidate_kalshi_ledger(PRED, RESULT, COLUMNS, side_field="lado", cancelled_match_ids=postponed_match_ids())
