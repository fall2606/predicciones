"""MLB/Kalshi forecasts and append-only prediction settlement ledger."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import math
import os
import re
import time
import unicodedata
import requests
import numpy as np
from scipy.stats import nbinom

API_MLB = "https://statsapi.mlb.com/api/v1"
API_KALSHI = "https://external-api.kalshi.com/trade-api/v2"
ROOT = Path(__file__).resolve().parent
PRED_FILE = ROOT / "predicciones_mlb.csv"
RESULT_FILE = ROOT / "resultados_mlb.csv"
TZ = "America/Chicago"
TIMEOUT = 25

PRED_COLUMNS = [
    "prediction_id", "event_ticker", "ticker", "match_id", "fecha", "local", "visita",
    "categoria", "mercado", "lado", "probabilidad_modelo", "precio_captura",
    "probabilidad_implicita", "edge", "estado_modelo", "reglas", "capturado_en",
]
RESULT_COLUMNS = ["prediction_id", "estado", "resultado_kalshi", "marcador", "resuelto_en"]


def get_json(url, params=None):
    last = None
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                last = requests.HTTPError(f"HTTP {r.status_code} en {url}", response=r)
                if attempt < 5:
                    time.sleep(min(2 ** attempt, 30))
                    continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last = exc
            if attempt == 5:
                break
            time.sleep(min(2 ** attempt, 30))
    raise last


def kalshi(path, **params):
    return get_json(f"{API_KALSHI}/{path}", params)


def normalizar(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def leer_csv(path, columnas):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def append_csv(path, columns, rows):
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writerows(rows)


def cargar_calendario(inicio, fin):
    data = get_json(f"{API_MLB}/schedule", {
        "sportId": 1, "startDate": inicio, "endDate": fin,
        "gameTypes": "R,F,D,L,W", "hydrate": "probablePitcher",
    })
    return [g for d in data.get("dates", []) for g in d.get("games", [])]


def cargar_stats(season):
    data = get_json(f"{API_MLB}/teams/stats", {
        "stats": "season", "group": "hitting", "season": season, "sportIds": 1,
    })
    stats = {}
    for block in data.get("stats", []):
        for split in block.get("splits", []):
            team, stat = split.get("team", {}), split.get("stat", {})
            n, runs = float(stat.get("gamesPlayed") or 0), float(stat.get("runs") or 0)
            if team.get("id") and n:
                stats[int(team["id"])] = {"name": team.get("name", ""), "rpg": runs/n, "games": n}
    return stats


def modelo_partido(game, stats):
    h = game.get("teams", {}).get("home", {})
    a = game.get("teams", {}).get("away", {})
    hid, aid = (h.get("team") or {}).get("id"), (a.get("team") or {}).get("id")
    if not hid or not aid:
        return None
    mean = float(np.mean([x["rpg"] for x in stats.values()])) if stats else 4.4
    hs, aw = stats.get(int(hid)), stats.get(int(aid))
    mh = (0.75 * hs["rpg"] + 0.25 * mean) if hs else mean
    ma = (0.75 * aw["rpg"] + 0.25 * mean) if aw else mean
    mh, ma = max(1.8, mh * 1.03), max(1.8, ma * 0.98)
    axis = np.arange(0, 26)
    def pmf(mu):
        dispersion = 0.12
        n = 1 / dispersion
        return nbinom.pmf(axis, n, n/(n+mu))
    matrix = np.outer(pmf(mh), pmf(ma))
    matrix /= matrix.sum()
    return {
        "id": str(game.get("gamePk", "")),
        "home": h.get("team", {}).get("name", "Local"),
        "away": a.get("team", {}).get("name", "Visitante"),
        "date": game.get("gameDate", ""),
        "status": (game.get("status") or {}).get("abstractGameState", ""),
        "runs_home": h.get("score"), "runs_away": a.get("score"),
        "mu_home": mh, "mu_away": ma, "axis": axis, "matrix": matrix,
    }


def partidos_proximos():
    now = datetime.now(timezone.utc)
    start = now.date() - timedelta(days=1)
    end = now.date() + timedelta(days=3)
    stats = cargar_stats(now.year)
    games = cargar_calendario(start.isoformat(), end.isoformat())
    return [m for g in games if (m := modelo_partido(g, stats))], stats


def clasificar(serie, market):
    text = normalizar(" ".join(str(market.get(k, "")) for k in ("title", "yes_sub_title", "subtitle", "rules_primary")))
    s = normalizar(serie)
    if (any(x in s for x in ("player", "strikeout", "home_run", "hits", "total_bases", "rbi", "pitcher", "kxmlbhr", "kxmlbtb", "kxmlbrbi", "kxmlbsb", "kxmlbouts", "kxmlber", "kxmlbwalk"))
            or any(x in text for x in ("home run", "total bases", "strikeout", "hits", "rbis", "walks", "outs recorded", "earned runs", "stolen bases"))):
        return "Props de jugador"
    if any(x in text for x in ("first inning", "1st inning", "yrfi", "nrfi")) or "firstinning" in s:
        return "1.ª entrada / YRFI-NRFI"
    if any(x in text for x in ("first 5", "first five", "first 3", "first three", "first 7", "first seven")) or any(x in s for x in ("first5", "first3", "first7")):
        return "Entradas parciales"
    if "team total" in text or "teamtotal" in s:
        return "Total por equipo"
    if "spread" in s or "run line" in text or "runline" in s:
        return "Run line / margen"
    if "total" in s or re.search(r"\b(over|under|more than|less than)\b", text):
        return "Total de carreras"
    if "inning" in s or "inning" in text:
        return "Entrada individual"
    if "exact" in text and ("score" in text or "runs" in text):
        return "Marcador exacto"
    if "extra inning" in text or "extra innings" in text:
        return "Entradas extra"
    if any(x in s for x in ("game", "moneyline", "winner")) or "win" in text:
        return "Ganador (moneyline)"
    return "Otros mercados MLB"


def mercado_probabilidad(market, game):
    """Estimate game-level contracts; leave unsupported player props explicitly unmodelled."""
    title = str(market.get("title") or market.get("yes_sub_title") or "")
    text = normalizar(" ".join((title, str(market.get("subtitle") or ""), str(market.get("yes_sub_title") or ""))))
    series = normalizar(market.get("series_ticker", ""))
    mat, axis = game["matrix"], game["axis"]
    h, a = game["home"], game["away"]
    home = normalizar(h) in text
    away = normalizar(a) in text
    runs = re.search(r"(?:over|under|more than|less than)\s*\$?([0-9]+(?:\.[0-9]+)?)", text)
    # Tickers sometimes encode a numeric line while the market title does not.
    ticker = str(market.get("ticker", ""))
    line_match = re.search(r"(?:TOTAL|OVER|UNDER|RUNS|SPREAD)[A-Z_-]*([0-9]+(?:P[0-9]+)?)", ticker, re.I)
    line = float(runs.group(1)) if runs else (float(line_match.group(1).replace("P", ".")) if line_match else None)
    category = clasificar(market.get("series_ticker", ""), market)
    p_yes = None
    if category == "Ganador (moneyline)":
        if home:
            p_yes = float(mat[np.tril_indices_from(mat, -1)].sum())
        elif away:
            p_yes = float(mat[np.triu_indices_from(mat, 1)].sum())
    elif category == "Total de carreras" and line is not None:
        total = axis[:, None] + axis[None, :]
        if "under" in text or "less than" in text:
            p_yes = float(mat[total < line].sum())
        elif "over" in text or "more than" in text:
            p_yes = float(mat[total > line].sum())
    elif category == "Total por equipo" and line is not None and (home or away):
        scores = axis[:, None] if home else axis[None, :]
        if "under" in text or "less than" in text:
            p_yes = float(mat[scores < line].sum())
        elif "over" in text or "more than" in text:
            p_yes = float(mat[scores > line].sum())
    elif category == "Run line / margen" and line is not None and (home or away):
        margin = axis[:, None] - axis[None, :] if home else axis[None, :] - axis[:, None]
        if "under" in text or "plus" in text or "+" in title:
            p_yes = float(mat[margin > -line].sum())
        else:
            p_yes = float(mat[margin > line].sum())
    elif category == "1.ª entrada / YRFI-NRFI":
        p_any_run = 1 - math.exp(-(game["mu_home"] + game["mu_away"]) / 9)
        p_yes = p_any_run if any(x in text for x in ("yrfi", "yes run", "at least one run", "a run scored")) else 1-p_any_run
    if p_yes is not None:
        return min(max(p_yes, 0.001), 0.999), "modelado"
    return None, "sin_modelo_especifico"


def series_mlb():
    all_series, cursor = [], None
    for _ in range(10):
        params = {"limit": 200, "category": "Sports"}
        if cursor:
            params["cursor"] = cursor
        page = kalshi("series", **params)
        all_series.extend(page.get("series", []))
        cursor = page.get("cursor")
        if not cursor:
            break
    selected = []
    for s in all_series:
        ticker = str(s.get("ticker", "")).upper()
        name = normalizar(s.get("title") or s.get("name") or "")
        tags = " ".join(map(str, s.get("tags") or [])).lower()
        excluded = ("FODT", "FUTURE", "DIVISION", "PLAYOFF", "CHAMPION", "PENNANT", "MVP", "AWARD", "TEAMWINS", "WORLD SERIES")
        if (ticker.startswith(("KXMLB", "KXBASEBALL")) or ("mlb" in name and ("baseball" in tags or "sports" in tags)) or ("major league baseball" in name)) and not any(x in ticker.upper() for x in excluded):
            selected.append(ticker)
    return sorted(set(selected))


def eventos_mlb(series=None):
    events = []
    for ticker in (series if series is not None else series_mlb()):
        cursor = None
        for _ in range(20):
            params = {"limit": 200, "series_ticker": ticker, "status": "open", "with_nested_markets": "true"}
            if cursor:
                params["cursor"] = cursor
            try:
                page = kalshi("events", **params)
            except requests.RequestException as exc:
                print(f"No se pudo leer la serie MLB {ticker}: {exc}")
                break
            events.extend(page.get("events", []))
            cursor = page.get("cursor")
            time.sleep(0.6)
            if not cursor:
                break
    return events


def encontrar_partido(event, games):
    title = normalizar(event.get("title") or event.get("sub_title") or "")
    event_date = str(event.get("strike_date") or event.get("expected_expiration_time") or event.get("start_time") or "")[:10]
    ranked = []
    for g in games:
        def aparece(nombre):
            nombre = normalizar(nombre)
            ultima = nombre.split()[-1] if nombre else ""
            return bool(nombre and nombre in title) or bool(len(ultima) >= 4 and re.search(r"\b" + re.escape(ultima) + r"\b", title))
        score = int(aparece(g["home"])) + int(aparece(g["away"]))
        if score == 2:
            try:
                delta = abs((datetime.fromisoformat(g["date"].replace("Z", "+00:00")).date()
                             - datetime.fromisoformat(event_date).date()).days) if event_date else 0
            except ValueError:
                delta = 0
            ranked.append((delta, g))
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1] if ranked else None


def mercados_liquidados(series):
    """Consulta por serie para reducir llamadas y respetar los límites de Kalshi."""
    finales = {}
    for serie in series:
        cursor = None
        for _ in range(30):
            params = {"limit": 200, "series_ticker": serie, "status": "settled"}
            if cursor:
                params["cursor"] = cursor
            try:
                page = kalshi("markets", **params)
            except requests.RequestException as exc:
                print(f"No se pudieron consultar liquidaciones de {serie}: {exc}")
                break
            for market in page.get("markets", []):
                ticker = market.get("ticker")
                result = str(market.get("result", "")).lower()
                if ticker and result in {"yes", "no"}:
                    finales[ticker] = result
            cursor = page.get("cursor")
            if not cursor:
                break
            time.sleep(0.35)
        time.sleep(0.5)
    return finales


def main():
    now = datetime.now(timezone.utc)
    games, _ = partidos_proximos()
    series = series_mlb()
    events = eventos_mlb(series)
    existing = leer_csv(PRED_FILE, PRED_COLUMNS)
    settled = leer_csv(RESULT_FILE, RESULT_COLUMNS)
    existing_ids = {x["prediction_id"] for x in existing}
    settled_ids = {x["prediction_id"] for x in settled}
    captured = now.isoformat()
    new_rows = []
    event_count = 0
    for event in events:
        markets = event.get("markets") or []
        if not markets:
            # Nested markets may be omitted for some event endpoints; fetch them separately.
            ticker = event.get("event_ticker")
            if ticker:
                try:
                    markets = kalshi("markets", event_ticker=ticker, limit=200).get("markets", [])
                except requests.RequestException:
                    markets = []
        if not markets:
            continue
        game = encontrar_partido(event, games)
        event_count += 1
        for m in markets:
            if m.get("status") not in (None, "active", "open"):
                continue
            ticker = str(m.get("ticker") or "")
            if not ticker:
                continue
            m["series_ticker"] = m.get("series_ticker") or event.get("series_ticker", "")
            category = clasificar(m["series_ticker"], m)
            p_yes, model_state = mercado_probabilidad(m, game) if game else (None, "partido_no_identificado")
            yes_price = _price(m, "yes_ask_dollars", "yes_bid_dollars")
            no_price = _price(m, "no_ask_dollars", "no_bid_dollars")
            fecha = (game or {}).get("date") or event.get("start_time") or event.get("expected_expiration_time") or ""
            for side, price, probability in (
                ("YES", yes_price, p_yes),
                ("NO", no_price, 1-p_yes if p_yes is not None else None),
            ):
                pid = f"{ticker}|{side}"
                if pid in existing_ids:
                    continue
                implied = price
                edge = probability - price if probability is not None and price is not None else None
                new_rows.append({
                    "prediction_id": pid, "event_ticker": event.get("event_ticker", ""),
                    "ticker": ticker, "match_id": (game or {}).get("id", ""),
                    "fecha": fecha, "local": (game or {}).get("home", ""),
                    "visita": (game or {}).get("away", ""), "categoria": category,
                    "mercado": m.get("yes_sub_title") or m.get("title") or ticker,
                    "lado": side, "probabilidad_modelo": f"{probability:.6f}" if probability is not None else "",
                    "precio_captura": f"{price:.6f}" if price is not None else "",
                    "probabilidad_implicita": f"{implied:.6f}" if implied is not None else "",
                    "edge": f"{edge:.6f}" if edge is not None else "",
                    "estado_modelo": model_state,
                    "reglas": m.get("rules_primary") or m.get("rules_secondary") or "",
                    "capturado_en": captured,
                })
                existing_ids.add(pid)

    append_csv(PRED_FILE, PRED_COLUMNS, new_rows)
    # Resolve only from Kalshi's final contract result; never rewrite an existing row.
    existing = leer_csv(PRED_FILE, PRED_COLUMNS)
    settled = leer_csv(RESULT_FILE, RESULT_COLUMNS)
    settled_ids = {x["prediction_id"] for x in settled}
    final_results = mercados_liquidados(series)
    result_rows = []
    for row in existing:
        pid = row["prediction_id"]
        if pid in settled_ids or not row.get("ticker"):
            continue
        result = final_results.get(row["ticker"], "")
        if result not in {"yes", "no"}:
            continue
        # Only Kalshi's official final result can settle a contract; never guess.
        won = (result == "yes") == (row["lado"] == "YES")
        game = next((g for g in games if g["id"] == row.get("match_id")), None)
        scoreboard = ""
        if game and game.get("runs_home") is not None and game.get("runs_away") is not None:
            scoreboard = f'{game["runs_away"]}-{game["runs_home"]}'
        result_rows.append({
            "prediction_id": pid, "estado": "WIN" if won else "LOSS",
            "resultado_kalshi": result.upper(), "marcador": scoreboard,
            "resuelto_en": datetime.now(timezone.utc).isoformat(),
        })
        settled_ids.add(pid)
    append_csv(RESULT_FILE, RESULT_COLUMNS, result_rows)
    print(f"Eventos MLB: {event_count} | Nuevas predicciones inmutables: {len(new_rows)} | WIN/LOSS añadidos: {len(result_rows)}")
    print("Mercados no modelados se conservan y se liquidan con el resultado oficial de Kalshi.")


def _price(market, *keys):
    for key in keys:
        value = market.get(key)
        try:
            if value not in (None, ""):
                p = float(value)
                if p > 1:
                    p /= 100
                if 0 < p < 1:
                    return p
        except (ValueError, TypeError):
            pass
    return None


if __name__ == "__main__":
    main()
