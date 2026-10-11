"""Reconcile terminal predictions against official completed-match calendars."""
import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SPORTS = {
    "LaLiga": ("predicciones_laliga.csv", "resultados_laliga.csv", "football", "PD"),
    "Champions": ("predicciones_champions.csv", "resultados_champions.csv", "football", "CL"),
    "MLB": ("predicciones_mlb.csv", "resultados_mlb.csv", "mlb", ""),
}
TERMINAL = {"WIN", "LOSS", "ANULADA"}


def rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def finished_ids(kind, competition):
    now = datetime.now(timezone.utc)
    if kind == "mlb":
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "startDate": (now.date() - timedelta(days=14)).isoformat(),
                    "endDate": now.date().isoformat(), "gameTypes": "R,F,D,L,W"},
            timeout=8,
        )
        response.raise_for_status()
        games = [g for day in response.json().get("dates", []) for g in day.get("games", [])]
        return {str(g["gamePk"]) for g in games
                if g.get("gamePk") is not None
                and str(g.get("status", {}).get("abstractGameState", "")).casefold() == "final"}
    token = os.getenv("CLAVE")
    if not token:
        raise RuntimeError("falta secreto CLAVE; no se puede comprobar calendario oficial")
    season = now.year if now.month >= 7 else now.year - 1
    response = requests.get(
        f"https://api.football-data.org/v4/competitions/{competition}/matches",
        headers={"X-Auth-Token": token}, params={"season": season}, timeout=8,
    )
    response.raise_for_status()
    matches = response.json().get("matches", [])
    return {str(m["id"]) for m in matches
            if m.get("id") is not None
            and str(m.get("status", "")).upper() in
            {"FINISHED", "AWARDED", "CANCELLED", "CANCELED", "POSTPONED"}}


def reconcile(name, pred_name, result_name, kind, competition):
    pred = rows(ROOT / pred_name)
    result = rows(ROOT / result_name)
    done = finished_ids(kind, competition)
    terminal_ids = {
        str(r.get("prediction_id", "")) for r in result
        if str(r.get("estado", "")).upper() in TERMINAL
    }
    finished_predictions = [
        p for p in pred
        if str(p.get("match_id", "")) in done and str(p.get("prediction_id", "")).strip()
    ]
    pending = [p for p in finished_predictions
               if str(p.get("prediction_id", "")) not in terminal_ids]
    registered = len(finished_predictions) - len(pending)
    print(f"CONSULTA {name}: SELECT prediction_id FROM {pred_name} "
          f"WHERE match_id IN (partidos_oficiales_finalizados) "
          f"AND prediction_id NOT IN (SELECT prediction_id FROM {result_name} "
          f"WHERE estado IN ('WIN','LOSS','ANULADA'));")
    print(f"RECONCILIACIÓN {name}: predicciones_partido_terminado={len(finished_predictions)} "
          f"| resultados_terminales_asociados={registered} | diferencia={len(pending)} "
          f"| pendientes={len(pending)}")
    if pending:
        for row in pending[:30]:
            print(f"PENDIENTE {name} · {row.get('prediction_id')} · match_id={row.get('match_id')} · ticker={row.get('ticker','')}")
        if len(pending) > 30:
            print(f"... y {len(pending)-30} pendientes adicionales")
    return len(pending)


def main():
    total_pending = 0
    for name, (pred, result, kind, competition) in SPORTS.items():
        try:
            total_pending += reconcile(name, pred, result, kind, competition)
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR RECONCILIACIÓN {name}: {exc}")
            total_pending += 1
    print(f"RECONCILIACIÓN TOTAL: pendientes={total_pending}")
    if total_pending:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
