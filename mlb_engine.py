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
from prediction_core import KALSHI_API_BASE, append_immutable_predictions, valid_probability, sports_series, events_for_series, rank_market_candidates, rank_single_candidates, extract_line, single_candidate_rejection_reasons
from mlb_math import score_probability_matrix

API_MLB = "https://statsapi.mlb.com/api/v1"
API_KALSHI = KALSHI_API_BASE
ROOT = Path(__file__).resolve().parent
PRED_FILE = ROOT / "predicciones_mlb.csv"
RESULT_FILE = ROOT / "resultados_mlb.csv"
TZ = "America/Chicago"
TIMEOUT = 15

MLB_TEAM_CODES = {
    "arizona diamondbacks": "ARI", "atlanta braves": "ATL", "baltimore orioles": "BAL",
    "boston red sox": "BOS", "chicago cubs": "CHC", "chicago white sox": "CWS",
    "cincinnati reds": "CIN", "cleveland guardians": "CLE", "colorado rockies": "COL",
    "detroit tigers": "DET", "houston astros": "HOU", "kansas city royals": "KC",
    "los angeles angels": "LAA", "los angeles dodgers": "LAD", "miami marlins": "MIA",
    "milwaukee brewers": "MIL", "minnesota twins": "MIN", "new york mets": "NYM",
    "new york yankees": "NYY", "athletics": "ATH", "oakland athletics": "ATH",
    "philadelphia phillies": "PHI", "pittsburgh pirates": "PIT", "san diego padres": "SD",
    "san francisco giants": "SF", "seattle mariners": "SEA", "st louis cardinals": "STL",
    "tampa bay rays": "TB", "texas rangers": "TEX", "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}


def codigo_equipo_mlb(nombre):
    return MLB_TEAM_CODES.get(normalizar(nombre), "")

PRED_COLUMNS = [
    "prediction_id", "event_ticker", "ticker", "match_id", "fecha", "local", "visita",
    "categoria", "mercado", "line", "lado", "seleccion", "bloque", "probabilidad_modelo", "precio_captura",
    "probabilidad_implicita", "edge", "estado_modelo", "reglas", "capturado_en", "disponibilidad",
]
RESULT_COLUMNS = ["prediction_id", "estado", "resultado_kalshi", "marcador", "resuelto_en"]


def get_json(url, params=None):
    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                last = requests.HTTPError(f"HTTP 429 (límite de peticiones) en {url}", response=r)
                if attempt < 3:
                    time.sleep(1 + attempt * 2)
                    continue
                raise last
            if r.status_code >= 500:
                last = requests.HTTPError(f"HTTP {r.status_code} en {url}", response=r)
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise last
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last = exc
            if attempt == 3:
                break
            time.sleep(1 + attempt * 2)
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
            if team.get("id") and math.isfinite(n) and math.isfinite(runs) and n > 0 and runs >= 0:
                stats[int(team["id"])] = {"name": team.get("name", ""), "rpg": runs/n, "games": n}
    return stats


def modelo_partido(game, stats):
    h = game.get("teams", {}).get("home", {})
    a = game.get("teams", {}).get("away", {})
    hid, aid = (h.get("team") or {}).get("id"), (a.get("team") or {}).get("id")
    if not hid or not aid:
        return None
    rates = [float(x["rpg"]) for x in stats.values() if math.isfinite(float(x.get("rpg", float("nan")))) and float(x.get("rpg", -1)) >= 0]
    mean = float(np.mean(rates)) if rates else 4.4
    if not math.isfinite(mean) or mean <= 0:
        mean = 4.4
    hs, aw = stats.get(int(hid)), stats.get(int(aid))
    mh = (0.75 * hs["rpg"] + 0.25 * mean) if hs else mean
    ma = (0.75 * aw["rpg"] + 0.25 * mean) if aw else mean
    mh, ma = max(1.8, mh * 1.03), max(1.8, ma * 0.98)
    axis = np.arange(0, 26)
    # Root cause of historical NaN%: extreme-but-finite API rates can underflow every
    # PMF value on the truncated 0..25 axis, making matrix.sum()==0 before normalization.
    # Use the shared bounded-mean/fallback normalizer instead of dividing by zero.
    matrix = score_probability_matrix(axis, mh, ma)
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
    # A generic inning total is not a full-game total; this model has no per-inning score distribution.
    if "inning" in s or "inning" in text:
        return "Entrada individual"
    if "team total" in text or "teamtotal" in s:
        return "Total por equipo"
    if "spread" in s or "run line" in text or "runline" in s:
        return "Run line / margen"
    if "total" in s or re.search(r"\b(over|under|more than|less than)\b", text):
        return "Total de carreras"
    if "exact" in text and ("score" in text or "runs" in text):
        return "Marcador exacto"
    if "extra inning" in text or "extra innings" in text:
        return "Entradas extra"
    if any(x in s for x in ("game", "moneyline", "winner")) or "win" in text:
        return "Ganador (moneyline)"
    return "Otros mercados MLB"


def mercado_probabilidad(market, game):
    """Model supported game, team, inning and exact-score contracts; never invent player-prop probabilities."""
    title = str(market.get("title") or market.get("yes_sub_title") or "")
    text = normalizar(" ".join((title, str(market.get("subtitle") or ""), str(market.get("yes_sub_title") or ""))))
    mat, axis = game["matrix"], game["axis"]
    h, a = game["home"], game["away"]
    home, away = normalizar(h) in text, normalizar(a) in text
    ticker = str(market.get("ticker", ""))
    category = clasificar(market.get("series_ticker", ""), market)
    home_code, away_code = codigo_equipo_mlb(h), codigo_equipo_mlb(a)
    contract_code = ticker.rsplit("-", 1)[-1].upper()
    if category in {"Ganador (moneyline)", "Total por equipo", "Run line / margen"} and contract_code in {home_code, away_code}:
        home, away = contract_code == home_code, contract_code == away_code

    # Prefer the readable contract line; tickers are only a fallback.
    runs = re.search(r"(?:over|under|more than|less than)\s*\$?([0-9]+(?:\.[0-9]+)?)", text)
    line_match = re.search(r"(?:TOTAL|OVER|UNDER|RUNS|SPREAD)[A-Z_-]*([0-9]+(?:P[0-9]+)?)", ticker, re.I)
    line = float(runs.group(1)) if runs else (float(line_match.group(1).replace("P", ".")) if line_match else None)
    p_yes = None
    model_state = "modelado"

    # First 3/5/7 innings: scale expected runs to the requested fraction of a game.
    if category == "Entradas parciales":
        if any(token in text for token in ("first 3", "first three")) or any(token in ticker.lower() for token in ("first3", "1st3")):
            fraction = 3 / 9
        elif any(token in text for token in ("first 7", "first seven")) or any(token in ticker.lower() for token in ("first7", "1st7")):
            fraction = 7 / 9
        else:
            fraction = 5 / 9
        partial_axis = np.arange(0, 18)
        dispersion = 0.12
        n = 1 / dispersion
        def pmf(mu):
            return nbinom.pmf(partial_axis, n, n / (n + max(0.05, mu * fraction)))
        mat = np.outer(pmf(game["mu_home"]), pmf(game["mu_away"]))
        mat /= mat.sum()
        axis = partial_axis
        # Normalize category so the common market predicates below can handle it.
        if any(word in text for word in ("moneyline", "winner", "to win", "win the first", "wins the first")):
            category = "Ganador (moneyline)"
        elif "spread" in text or "run line" in text:
            category = "Run line / margen"
        elif "team total" in text:
            category = "Total por equipo"
        else:
            category = "Total de carreras"

    if category == "Ganador (moneyline)":
        tie_probability = float(np.trace(mat))
        if home:
            p_yes = float(mat[np.tril_indices_from(mat, -1)].sum() + 0.5 * tie_probability)
        elif away:
            p_yes = float(mat[np.triu_indices_from(mat, 1)].sum() + 0.5 * tie_probability)
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
        p_yes = p_any_run if any(x in text for x in ("yrfi", "yes run", "at least one run", "a run scored")) else 1 - p_any_run
    elif category == "Entradas extra":
        # Extra innings are possible when the game is tied after nine; estimate from the score matrix.
        p_yes = float(np.trace(mat))
        if any(word in text for word in ("no extra innings", "no extra inning", "not go to extra")):
            p_yes = 1 - p_yes
    elif category == "Marcador exacto":
        # Support explicit "X-Y" score contracts when team order can be inferred from the title.
        score = re.search(r"\b([0-9]{1,2})\s*[-:]\s*([0-9]{1,2})\b", title)
        if score:
            x, y = int(score.group(1)), int(score.group(2))
            if home and away:
                p_yes = float(mat[x, y]) if x < mat.shape[0] and y < mat.shape[1] else 0.0
    if p_yes is not None and math.isfinite(p_yes):
        return min(max(float(p_yes), 0.001), 0.999), model_state
    return None, "sin_modelo_especifico"


def series_mlb():
    all_series = sports_series()
    selected = []
    for item in all_series:
        ticker = str(item.get("ticker", "")).upper()
        name = normalizar(item.get("title") or item.get("name") or "")
        tags = " ".join(map(str, item.get("tags") or [])).lower()
        excluded = ("FODT", "FUTURE", "DIVISION", "PLAYOFF", "CHAMPION", "PENNANT", "MVP", "AWARD", "TEAMWINS", "WORLD SERIES", "SERIES", "EXPAND", "INNINGTOTAL")
        if (ticker.startswith(("KXMLB", "KXBASEBALL")) or ("mlb" in name and ("baseball" in tags or "sports" in tags)) or ("major league baseball" in name)) and not any(x in ticker for x in excluded):
            selected.append(ticker)
    return sorted(set(selected))


def eventos_mlb(series=None):
    """Shared paginated Kalshi event/market catalogue for the selected MLB series."""
    return events_for_series(series if series is not None else series_mlb(), statuses=("open",))


def encontrar_partido(event, games):
    title = normalizar(event.get("title") or event.get("sub_title") or "")
    event_ticker = str(event.get("event_ticker") or "").upper()
    event_date = str(event.get("strike_date") or event.get("expected_expiration_time") or event.get("start_time") or "")[:10]
    ranked = []
    for g in games:
        def aparece(nombre):
            normalized = normalizar(nombre)
            code = codigo_equipo_mlb(nombre)
            if code and code in event_ticker:
                return True
            tokens = normalized.split()
            phrases = [normalized, " ".join(tokens[:2]) if len(tokens) >= 2 else "", tokens[0] if tokens else ""]
            ultima = tokens[-1] if tokens else ""
            return any(phrase and len(phrase) >= 4 and phrase in title for phrase in phrases) or bool(len(ultima) >= 4 and re.search(r"\b" + re.escape(ultima) + r"\b", title))
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
    existing_ids = {x["prediction_id"] for x in existing}
    captured = now.isoformat()
    by_match = {}
    event_count = 0

    # Model every actual, match-specific Kalshi market that this model supports.
    # Unmodelled contracts are deliberately excluded instead of being saved as NaN/blank picks.
    for event in events:
        markets = event.get("markets") or []
        if not markets and event.get("event_ticker"):
            try:
                markets = kalshi("markets", event_ticker=event["event_ticker"], limit=200).get("markets", [])
            except requests.RequestException:
                markets = []
        if not markets:
            continue
        game = encontrar_partido(event, games)
        if not game or game.get("status") != "Preview":
            continue
        event_count += 1
        for market in markets:
            if market.get("status") not in (None, "active", "open", "unopened"):
                continue
            ticker = str(market.get("ticker") or "")
            if not ticker:
                continue
            market["series_ticker"] = market.get("series_ticker") or event.get("series_ticker", "")
            category = clasificar(market["series_ticker"], market)
            p_yes, model_state = mercado_probabilidad(market, game)
            if valid_probability(p_yes) is None:
                continue
            yes_price = _price(market, "yes_ask_dollars")
            no_price = _price(market, "no_ask_dollars")
            kickoff = game.get("date") or event.get("start_time") or event.get("expected_expiration_time") or ""
            for side, price, probability in (("YES", yes_price, p_yes), ("NO", no_price, 1-p_yes)):
                if valid_probability(probability) is None:
                    continue
                edge = probability - price if price is not None else None
                row = {
                    "prediction_id": "",
                    "event_ticker": event.get("event_ticker", ""),
                    "ticker": ticker,
                    "match_id": str(game.get("id", "")),
                    "fecha": kickoff,
                    "local": game.get("home", ""),
                    "visita": game.get("away", ""),
                    "categoria": category,
                    "mercado": market.get("yes_sub_title") or market.get("title") or ticker,
                    "line": extract_line(market.get("yes_sub_title") or market.get("title") or ""),
                    "lado": side,
                    "seleccion": side,
                    "bloque": "",
                    "probabilidad_modelo": f"{probability:.8f}",
                    "precio_captura": f"{price:.6f}" if price is not None else "",
                    "disponibilidad": "Disponible en Kalshi" if price is not None else "Sin precio en Kalshi",
                    "probabilidad_implicita": f"{price:.6f}" if price is not None else "",
                    "edge": f"{edge:.6f}" if edge is not None else "",
                    "estado_modelo": model_state,
                    "reglas": market.get("rules_primary") or market.get("rules_secondary") or "",
                    "capturado_en": captured,
                }
                by_match.setdefault(str(game.get("id", "")), []).append(row)

    new_rows = []
    minimum_combo = float(os.getenv("PROBABILIDAD_MINIMA_COMBO", "0.65"))
    for match_id, rows in by_match.items():
        candidates = []
        for row in rows:
            p = valid_probability(row.get("probabilidad_modelo"))
            if p is None:
                continue
            try:
                edge = float(row["edge"]) if row.get("edge") not in (None, "") else None
            except (TypeError, ValueError):
                edge = None
            candidates.append({
                "_row": row,
                "_p": p,
                "_lado": row["lado"],
                "Ticker": row["ticker"],
                "Apuesta": f'{row["mercado"]} · {row["lado"]}',
                "Categoría resumen": row["categoria"],
                "Probabilidad del modelo": f"{p:.1%}",
                "Precio de compra": f'{float(row["precio_captura"]):.0%}' if row["precio_captura"] else "—",
                "Ventaja (edge)": edge,
                "Lado Kalshi": row["lado"],
            })

        # Singles: price cap, positive edge, trivial-line exclusion and family diversification.
        singles = rank_single_candidates(candidates, limit=5)
        if not singles:
            reasons = single_candidate_rejection_reasons(candidates)
            print(f\"SENCILLAS SIN CANDIDATAS · {match_id} · {rows[0].get('local', '')} vs {rows[0].get('visita', '')}: {reasons or {'sin candidatos modelables': len(candidates)}}\")

        high, value = rank_market_candidates(candidates, min_probability=minimum_combo, high_limit=5, value_limit=10)
        limits = {"SENCILLAS": 5, "ALTA_PROBABILIDAD": 5, "A_TENER_EN_CUENTA": 10}
        existing_match = [item for item in existing if str(item.get("match_id", "")) == str(match_id)]
        block_counts = {name: 0 for name in limits}
        existing_singles = []
        for item in existing_match:
            block_name = str(item.get("bloque", "") or "")
            if block_name not in limits:
                category_name = str(item.get("categoria", ""))
                block_name = next((name for name in limits if category_name.startswith(name + " · ")), "")
            if block_name == "SENCILLAS":
                existing_singles.append(item)
            elif block_name in limits:
                block_counts[block_name] += 1
        # Old, now-filtered singles must not occupy the slots forever; preserve them in the ledger
        # but only count those which still meet the current public display rules.
        block_counts["SENCILLAS"] = len(rank_single_candidates(existing_singles, limit=limits["SENCILLAS"]))

        for block, selected in (("SENCILLAS", singles), ("ALTA_PROBABILIDAD", high), ("A_TENER_EN_CUENTA", value)):
            count = block_counts[block]
            for candidate in selected:
                if count >= limits[block]:
                    break
                row = dict(candidate["_row"])
                side = candidate.get("_lado") or row["lado"]
                ticker = row["ticker"]
                row["prediction_id"] = f"{ticker}|{side}|{block}"
                row["categoria"] = f"{block} · {row['categoria']}"
                row["bloque"] = block
                if row["prediction_id"] in existing_ids:
                    continue
                new_rows.append(row)
                existing_ids.add(row["prediction_id"])
                count += 1

    capture_summary = append_immutable_predictions(
        PRED_FILE, PRED_COLUMNS, new_rows, probability_field="probabilidad_modelo"
    )
    print(f"Eventos MLB: {event_count} | capturas aceptadas: {capture_summary['accepted']} | "
          f"descartadas: {capture_summary['rejected']}. Los resultados solo los escribe liquidar_mlb.py.")



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
