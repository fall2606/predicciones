"""NFL Kalshi capture engine with sport-specific score model and immutable ledger."""
from __future__ import annotations
import math, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
import requests
from prediction_core import (KALSHI_API_BASE, append_immutable_predictions, sports_series,
                            events_for_series, rank_market_candidates, rank_single_candidates,
                            valid_probability)
from probability_calibration import calibrate_probability

ROOT = Path(__file__).resolve().parent
PRED_FILE = ROOT / "predicciones_nfl.csv"
RESULT_FILE = ROOT / "resultados_nfl.csv"
CURRENT_FILE = ROOT / "nfl_probabilidades_actuales.csv"
PRED_COLUMNS = ["prediction_id","event_ticker","ticker","match_id","fecha","local","visita","categoria",
                "mercado","line","lado","seleccion","bloque","probabilidad_modelo","probabilidad_calibrada",
                "precio_captura","probabilidad_implicita","edge","capturado_en","disponibilidad","calibracion","marcador_proyectado"]
RESULT_COLUMNS = ["prediction_id","estado","resultado_kalshi","marcador","resuelto_en"]
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
TIMEOUT = 15

def _json(url, params=None):
    r = requests.get(url, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()

def _team_stats(days=100):
    now = datetime.now(timezone.utc)
    scores = {}
    for offset in range(0, days, 7):
        day = (now - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            data = _json(ESPN, {"dates": day, "limit": 100})
        except (requests.RequestException, ValueError, TypeError):
            continue
        for event in data.get("events", []):
            comp = (event.get("competitions") or [{}])[0]
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue
            competitors = comp.get("competitors") or []
            if len(competitors) != 2:
                continue
            try:
                home = next(x for x in competitors if x.get("homeAway") == "home")
                away = next(x for x in competitors if x.get("homeAway") == "away")
                hs, aws = float(home["score"]), float(away["score"])
                hn = _norm(home.get("team", {}).get("displayName"))
                an = _norm(away.get("team", {}).get("displayName"))
            except (StopIteration, KeyError, TypeError, ValueError):
                continue
            if not hn or not an:
                continue
            scores.setdefault(hn, []).append((hs, aws))
            scores.setdefault(an, []).append((aws, hs))
    return {team: {"scored": mean(x[0] for x in rows[-8:]),
                   "allowed": mean(x[1] for x in rows[-8:]), "games": len(rows)}
            for team, rows in scores.items()}

def _match_team(name, stats):
    key = _norm(name)
    if key in stats:
        return key
    matches = [team for team in stats if key and (key in team or team in key or (key.split() and team.split() and key.split()[-1] == team.split()[-1]))]
    return max(matches, key=len) if len(matches) == 1 else None

def projected_score(home, away, stats):
    h, a = _match_team(home, stats), _match_team(away, stats)
    if not h or not a or stats[h]["games"] < 2 or stats[a]["games"] < 2:
        return None
    league_scored = mean(x["scored"] for x in stats.values())
    league_allowed = mean(x["allowed"] for x in stats.values())
    if league_scored <= 0 or league_allowed <= 0:
        return None
    hs = (stats[h]["scored"] + stats[a]["allowed"]) / 2 + 1.3
    aws = (stats[a]["scored"] + stats[h]["allowed"]) / 2
    return max(3, hs), max(3, aws)

def _normal_cdf(x, mu, sigma=13.5):
    return 0.5 * (1 + math.erf((x-mu)/(sigma*math.sqrt(2))))

def _market_probability(market, home, away, home_score, away_score):
    title = str(market.get("yes_sub_title") or market.get("title") or "")
    ticker = str(market.get("ticker") or "").upper()
    text = (title + " " + ticker).lower()
    margin = home_score - away_score
    total = home_score + away_score
    # Winner/team markets: only match explicit team name, avoid interpreting arbitrary player props.
    if any(x in text for x in ("winner", "win", "moneyline", "to beat", "victory", "game")):
        team = home if _norm(home) in _norm(title) else away if _norm(away) in _norm(title) else None
        if team:
            p_home = _normal_cdf(margin, 0, 13.5)
            return p_home if team == home else 1-p_home, "Ganador del partido"
    line_match = re.search(r"(?<![a-z])([+-]?\d+(?:\.\d+)?)", title)
    if line_match:
        line = float(line_match.group(1))
        if "total" in text or "over" in text or "under" in text:
            over = _normal_cdf(total, line, 10.5)
            if "under" in text or "below" in text:
                over = 1-over
            return over, "Total de puntos"
        if "spread" in text or "handicap" in text or "point" in text:
            team = home if _norm(home) in _norm(title) else away if _norm(away) in _norm(title) else None
            if team:
                p = _normal_cdf(margin if team == home else -margin, -line, 13.5)
                return p, "Hándicap de puntos"
    return None, ""

def _price(market, side):
    keys = (f"{side.lower()}_ask_dollars", f"{side.lower()}_ask", f"{side.lower()}_bid_dollars", f"{side.lower()}_bid")
    for key in keys:
        value = market.get(key)
        try:
            p = float(value)
            if p > 1: p /= 100
            if 0 < p < 1: return p
        except (TypeError, ValueError):
            pass
    return None

def build_candidates(events, stats):
    candidates = []
    now = datetime.now(timezone.utc)
    for event in events:
        title = str(event.get("title") or "")
        # Event title is normally "Team A vs Team B"; don't guess when names cannot be separated.
        parts = re.split(r"\s+(?:vs\.?|versus|@)\s+", title, maxsplit=1, flags=re.I)
        if len(parts) != 2:
            continue
        home, away = parts[0].strip(), parts[1].strip()
        start = str(event.get("start_time") or event.get("strike_date") or "")
        try:
            kickoff = datetime.fromisoformat(start.replace("Z", "+00:00"))
            if kickoff.tzinfo is None: kickoff = kickoff.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if kickoff <= now or kickoff > now + timedelta(days=14):
            continue
        projected = projected_score(home, away, stats)
        if not projected:
            continue
        hs, aws = projected
        for market in event.get("markets") or []:
            ticker = str(market.get("ticker") or "")
            if not ticker or str(market.get("status") or "open").lower() not in {"open", "active"}:
                continue
            p_yes, category = _market_probability(market, home, away, hs, aws)
            if p_yes is None or not 0.01 < p_yes < 0.99:
                continue
            title_market = str(market.get("yes_sub_title") or market.get("title") or ticker)
            for side, probability in (("YES", p_yes), ("NO", 1-p_yes)):
                price = _price(market, side)
                edge = probability-price if price is not None else None
                candidates.append({
                    "event_ticker": str(event.get("event_ticker") or ""),
                    "ticker": ticker, "match_id": str(event.get("event_ticker") or ""),
                    "fecha": kickoff.astimezone(timezone.utc).isoformat(), "local": home, "visita": away,
                    "categoria": category, "mercado": title_market, "line": "",
                    "lado": side, "seleccion": side, "probabilidad_modelo": probability,
                    "precio_captura": price, "probabilidad_implicita": price, "edge": edge,
                    "disponibilidad": "Con precio" if price is not None else "Sin precio en Kalshi",
                    "marcador_proyectado": f"{hs:.1f}-{aws:.1f}",
                })
    return candidates

def main():
    series = sports_series(("KXNFL",))
    if not series:
        raise RuntimeError("No se encontró ninguna serie NFL KXNFL en el catálogo público de Kalshi.")
    events = events_for_series([x["ticker"] for x in series], statuses=("open",))
    stats = _team_stats()
    candidates = build_candidates(events, stats)
    # Keep snapshots immutable; use market + side + event identity to avoid duplicates.
    existing = set()
    if PRED_FILE.exists():
        import csv
        with PRED_FILE.open(newline="", encoding="utf-8-sig") as f:
            existing = {r.get("prediction_id","") for r in csv.DictReader(f)}
    captured = datetime.now(timezone.utc).isoformat()
    rows = []
    for row in candidates:
        category = row["categoria"]
        calibrated, note = calibrate_probability(row["probabilidad_modelo"], category, PRED_FILE, RESULT_FILE)
        row["probabilidad_calibrada"] = calibrated
        row["calibracion"] = note
        price = row["precio_captura"]
        row["edge"] = calibrated-price if price is not None else None
        row["bloque"] = ""
        row["capturado_en"] = captured
        row["prediction_id"] = f'{row["event_ticker"]}|{row["ticker"]}|{row["lado"]}'
        if row["prediction_id"] not in existing:
            rows.append(row)
    priced = [r for r in rows if r.get("precio_captura") is not None]
    ranking_rows = [dict(r, _p=r["probabilidad_calibrada"], **{"Ventaja (edge)": r["edge"], "Apuesta sencilla": r["mercado"]}) for r in priced]
    high, value = rank_market_candidates(ranking_rows, min_probability=0.65, high_limit=5, value_limit=10)
    high_ids = {x["prediction_id"] for x in high}
    value_ids = {x["prediction_id"] for x in value}
    singles = rank_single_candidates(ranking_rows, limit=5)
    single_ids = {x["prediction_id"] for x in singles}
    # Current snapshot is replaceable; prediction history below remains append-only.
    from csv import DictWriter
    current_fields = ["event_ticker","ticker","local","visita","fecha","categoria","mercado","lado","probabilidad_modelo","probabilidad_calibrada","precio_captura","edge","calibracion","marcador_proyectado"]
    with CURRENT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=current_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    for row in rows:
        row["bloque"] = ("SENCILLAS" if row["prediction_id"] in single_ids else
                         "ALTA_PROBABILIDAD" if row["prediction_id"] in high_ids else
                         "A_TENER_EN_CUENTA" if row["prediction_id"] in value_ids else "DETALLE")
    append_immutable_predictions(PRED_FILE, PRED_COLUMNS, rows)
    print(f"NFL: eventos abiertos={len(events)}, candidatos modelados={len(candidates)}, capturas nuevas={len(rows)}")

if __name__ == "__main__":
    main()
