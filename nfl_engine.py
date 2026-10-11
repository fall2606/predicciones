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
EXCLUDED_FILE = ROOT / "nfl_predicciones_excluidas.csv"
EXCLUDED_COLUMNS = ["prediction_id", "motivo", "kickoff_real", "detectado_en"]
SCHEDULES = {}
PRED_COLUMNS = ["prediction_id","event_ticker","ticker","match_id","fecha","local","visita","categoria",
                "mercado","line","lado","seleccion","bloque","probabilidad_modelo","probabilidad_calibrada",
                "precio_captura","probabilidad_implicita","edge","capturado_en","disponibilidad","calibracion","marcador_proyectado"]
RESULT_COLUMNS = ["prediction_id","estado","resultado_kalshi","marcador","resuelto_en"]
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
TIMEOUT = 15
# All NFL rows captured before this validated pregame model release belong to the initial bad batch.
INITIAL_BAD_CAPTURE_CUTOFF = datetime(2026, 10, 11, 21, 55, tzinfo=timezone.utc)

def _json(url, params=None):
    r = requests.get(url, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()

def _team_stats(days=100):
    # ESPN season scoreboard supplies both completed results and scheduled kickoff times.
    now = datetime.now(timezone.utc)
    season_year = now.year if now.month >= 8 else now.year - 1
    scores = {}
    SCHEDULES.clear()
    season_types = [2, 3] if now.month in (1, 2) else [2]
    for season_type in season_types:
        try:
            data = _json(ESPN, {"dates": str(season_year), "seasontype": season_type, "limit": 1000})
        except (requests.RequestException, ValueError, TypeError):
            continue
        for event in data.get("events", []):
            competitors = (event.get("competitions") or [{}])[0].get("competitors") or []
            if len(competitors) != 2:
                continue
            try:
                home = next(x for x in competitors if x.get("homeAway") == "home")
                away = next(x for x in competitors if x.get("homeAway") == "away")
                hn = _norm(home.get("team", {}).get("displayName") or home.get("team", {}).get("name"))
                an = _norm(away.get("team", {}).get("displayName") or away.get("team", {}).get("name"))
                game_date = str(event.get("date") or "")
                if hn and an and game_date:
                    SCHEDULES.setdefault(tuple(sorted((hn, an))), []).append(game_date)
                if not (event.get("competitions") or [{}])[0].get("status", {}).get("type", {}).get("completed"):
                    continue
                hs, aws = float(home["score"]), float(away["score"])
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

def _team_in_market(title, home, away):
    text = _norm(title).split()
    home_tokens = _norm(home).split()
    away_tokens = _norm(away).split()
    home_hit = bool(home_tokens and home_tokens[-1] in text)
    away_hit = bool(away_tokens and away_tokens[-1] in text)
    if home_hit and not away_hit:
        return home
    if away_hit and not home_hit:
        return away
    if _norm(home) in _norm(title) and _norm(away) not in _norm(title):
        return home
    if _norm(away) in _norm(title) and _norm(home) not in _norm(title):
        return away
    return None

def _market_probability(market, home, away, home_score, away_score):
    title = str(market.get("yes_sub_title") or market.get("title") or "")
    full_text = " ".join(x for x in (title, str(market.get("title") or "")) if x)
    ticker = str(market.get("ticker") or "").upper()
    text = (full_text + " " + ticker).lower()
    # Only full-game markets are modeled; period and quarter contracts need separate score distributions.
    if any(term in text for term in ("1st half", "first half", "2nd half", "second half",
                                     "1st quarter", "2nd quarter", "3rd quarter", "4th quarter",
                                     "1h spread", "2h spread", "quarter spread")):
        return None, ""
    if re.search(r"KXNFL(?:1H|2H|1Q|2Q|3Q|4Q|Q1|Q2|Q3|Q4)", ticker):
        return None, ""
    margin = home_score - away_score
    total = home_score + away_score
    line_match = re.search(r"(?<![a-z])([+-]?\d+(?:\.\d+)?)", full_text)
    line = float(line_match.group(1)) if line_match else None

    is_total = "TOTAL" in ticker or "total" in text
    if "TEAMTOTAL" in ticker and line is not None:
        team = _team_in_market(title, home, away)
        if not team:
            return None, ""
        projected_team_total = home_score if team == home else away_score
        over = _normal_cdf(projected_team_total, line, 7.5)
        if "under" in title.lower() or "below" in title.lower():
            over = 1 - over
        return over, "Total de puntos del equipo"
    if is_total and line is not None:
        over = _normal_cdf(total, line, 10.5)
        if "under" in title.lower() or "below" in title.lower():
            over = 1 - over
        return over, "Total de puntos"

    is_spread = any(term in text for term in ("spread", "handicap", "wins by", "win by"))
    if is_spread and line is not None:
        team = _team_in_market(title, home, away)
        if team:
            # "wins by over X" uses a positive winning margin threshold; signed spreads
            # use the opposite sign of the quoted handicap.
            threshold = line if ("wins by over" in text or "win by over" in text or "wins by more than" in text) else -line
            projected_margin = margin if team == home else -margin
            return _normal_cdf(projected_margin, threshold, 13.5), "Hándicap de puntos"

    if any(term in text for term in ("winner", "moneyline", "to beat", "victory", "game winner")) or re.search(r"\bwin(?:s)?\b", title.lower()):
        team = _team_in_market(title, home, away)
        if team:
            p_home = _normal_cdf(margin, 0, 13.5)
            return p_home if team == home else 1 - p_home, "Ganador del partido"
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

def _parse_datetime(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

def _ticker_date(value):
    match = re.search(r"-(\d{2}[A-Z]{3}\d{2})", str(value or "").upper())
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%y%b%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None

def _scheduled_kickoff(home, away, stats, ticker_or_date=None):
    h, a = _match_team(home, stats), _match_team(away, stats)
    if not h or not a:
        return None
    values = SCHEDULES.get(tuple(sorted((h, a))), [])
    if isinstance(values, str):
        values = [values]
    candidates = [dt for dt in (_parse_datetime(value) for value in values) if dt is not None]
    if not candidates:
        return None
    target = _parse_datetime(ticker_or_date) if ticker_or_date else None
    if target is None:
        target = _ticker_date(ticker_or_date)
    if target is not None:
        return min(candidates, key=lambda dt: abs((dt-target).total_seconds()))
    now = datetime.now(timezone.utc)
    future = [dt for dt in candidates if dt > now]
    return min(future, key=lambda dt: dt) if future else max(candidates)

def _event_start(event):
    for field in ("start_time", "strike_date", "start_date"):
        value = event.get(field)
        if value:
            return str(value)
    # For grouping only: expected_expiration_time is usually shared by all markets
    # for the same game. It is NOT used as the actual kickoff.
    for market in event.get("markets") or []:
        value = market.get("expected_expiration_time")
        if value:
            return str(value)
    return ""


def merge_game_events(events):
    """Merge per-market Kalshi events into one fixture, retaining every real market ticker."""
    grouped = {}
    for source in events or []:
        event = dict(source)
        fixture = str(event.get("title") or "").split(":", 1)[0].strip()
        start = _event_start(event)
        teams = re.split(r"\s+(?:vs\.?|versus|@|at)\s+", fixture, maxsplit=1, flags=re.I)
        if len(teams) != 2:
            continue
        key = (_norm(fixture), start[:10])
        if key not in grouped:
            event["title"] = fixture
            event["markets"] = []
            grouped[key] = event
        current_event_ticker = str(grouped[key].get("event_ticker") or "").upper()
        source_event_ticker = str(source.get("event_ticker") or "").upper()
        period_pattern = r"KXNFL(?:1H|2H|1Q|2Q|3Q|4Q|Q1|Q2|Q3|Q4)"
        if re.search(period_pattern, current_event_ticker) and source_event_ticker and not re.search(period_pattern, source_event_ticker):
            grouped[key]["event_ticker"] = source.get("event_ticker")
        known = {str(m.get("ticker") or "") for m in grouped[key]["markets"]}
        for market in source.get("markets") or []:
            ticker = str(market.get("ticker") or "")
            if ticker and ticker not in known:
                grouped[key]["markets"].append(market)
                known.add(ticker)
    return list(grouped.values())

def _read_rows(path):
    import csv
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def reconcile_exclusions(stats):
    """Keep a canonical quarantine ledger for invalid historical captures."""
    existing_rows = _read_rows(EXCLUDED_FILE)
    existing_by_id = {str(row.get("prediction_id") or "").strip(): row for row in existing_rows}
    for row in _read_rows(PRED_FILE):
        pid = str(row.get("prediction_id") or "").strip()
        if not pid:
            continue
        event_ticker = str(row.get("event_ticker") or "").upper()
        market_ticker = str(row.get("ticker") or "").upper()
        period_pattern = r"KXNFL(?:1H|2H|1Q|2Q|3Q|4Q|Q1|Q2|Q3|Q4)"
        event_is_period = bool(re.search(period_pattern, event_ticker))
        market_is_period = bool(re.search(period_pattern, market_ticker))
        kickoff = _scheduled_kickoff(row.get("local"), row.get("visita"), stats, row.get("event_ticker") or row.get("ticker"))
        captured = _parse_datetime(row.get("capturado_en"))
        stored_kickoff = _parse_datetime(row.get("fecha"))
        late_capture = kickoff is not None and captured is not None and captured >= kickoff
        misdated_capture = (kickoff is not None and stored_kickoff is not None
                            and abs((stored_kickoff - kickoff).total_seconds()) > 6 * 3600)
        initial_bad_batch = captured is not None and captured < INITIAL_BAD_CAPTURE_CUTOFF
        if not market_is_period and not event_is_period and not late_capture and not misdated_capture and not initial_bad_batch:
            continue
        if market_is_period:
            reason = "Contrato de periodo/cuarto no cubierto por el modelo NFL"
        elif event_is_period:
            reason = "Captura histórica asociada a una serie de periodo; excluida por el modelo inicial incorrecto"
        elif initial_bad_batch:
            reason = "Captura de la primera versión no validada; excluida de resultados y calibración"
        elif misdated_capture:
            reason = "Fecha histórica difiere del kickoff real en más de 6 horas; captura inicial inválida"
        else:
            reason = "Captura registrada a la hora del inicio real o después; excluida de calibración prepartido"
        previous = existing_by_id.get(pid)
        kickoff_value = kickoff.isoformat() if kickoff else ""
        if previous and previous.get("kickoff_real", "") == kickoff_value and previous.get("motivo", "") == reason:
            continue
        existing_by_id[pid] = {
            "prediction_id": pid,
            "motivo": reason,
            "kickoff_real": kickoff_value,
            "detectado_en": datetime.now(timezone.utc).isoformat(),
        }
    import csv
    with EXCLUDED_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXCLUDED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_by_id.values())
    return set(existing_by_id)


def build_candidates(events, stats):
    candidates = []
    now = datetime.now(timezone.utc)
    for event in events:
        title = str(event.get("title") or "").split(":", 1)[0].strip()
        # Event title is normally "Team A vs Team B"; don't guess when names cannot be separated.
        parts = re.split(r"\s+(?:vs\.?|versus|@|at)\s+", title, maxsplit=1, flags=re.I)
        if len(parts) != 2:
            continue
        home, away = parts[0].strip(), parts[1].strip()
        start = event.get("start_time") or event.get("strike_date") or event.get("start_date")
        kickoff = _parse_datetime(start) if start else _scheduled_kickoff(home, away, stats, event.get("event_ticker"))
        if kickoff is None:
            continue
        if kickoff <= now or kickoff > now + timedelta(days=14):
            continue
        projected = projected_score(home, away, stats)
        if not projected:
            continue
        hs, aws = projected
        for market in event.get("markets") or []:
            ticker = str(market.get("ticker") or "").upper()
            event_ticker = str(event.get("event_ticker") or "").upper()
            # A grouped fixture may contain markets fetched from several Kalshi series.
            # Never score period/quarter contracts or relabel a period event as full game.
            period_pattern = r"KXNFL(?:1H|2H|1Q|2Q|3Q|4Q|Q1|Q2|Q3|Q4)"
            if (not ticker or re.search(period_pattern, ticker) or re.search(period_pattern, event_ticker)
                    or str(market.get("status") or "open").lower() not in {"open", "active"}):
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

def _rank_nfl_game(rows):
    """NFL-specific diversified ranking: one side per contract and per-game limits."""
    by_ticker = {}
    for row in rows or []:
        ticker = str(row.get("ticker") or "").strip()
        try:
            p = float(row.get("probabilidad_calibrada"))
            price = float(row.get("precio_captura"))
            edge = float(row.get("edge"))
        except (TypeError, ValueError):
            continue
        if not ticker or not (0 < p < 1) or not (0 < price < 1) or not math.isfinite(edge):
            continue
        previous = by_ticker.get(ticker)
        if previous is None or edge > float(previous.get("edge") or -999):
            by_ticker[ticker] = row
    eligible = list(by_ticker.values())

    def choose(predicate, limit, sort_key, per_category, exclude_ids=()):
        picked, counts = [], {}
        excluded = set(exclude_ids)
        for row in sorted((x for x in eligible if predicate(x) and x["prediction_id"] not in excluded), key=sort_key, reverse=True):
            category = str(row.get("categoria") or "").split(" · ", 1)[0]
            if counts.get(category, 0) >= per_category:
                continue
            picked.append(row["prediction_id"])
            counts[category] = counts.get(category, 0) + 1
            if len(picked) >= limit:
                break
        return set(picked)

    singles = choose(
        lambda r: float(r["precio_captura"]) <= 0.88 and float(r["probabilidad_calibrada"]) >= 0.55 and float(r["edge"]) >= 0.05,
        5, lambda r: (float(r["edge"]), float(r["probabilidad_calibrada"])), 2,
    )
    combos = choose(
        lambda r: float(r["precio_captura"]) <= 0.90 and float(r["probabilidad_calibrada"]) >= 0.65 and float(r["edge"]) >= 0.02,
        5, lambda r: (float(r["probabilidad_calibrada"]), float(r["edge"])), 2, singles,
    )
    value = choose(
        lambda r: float(r["edge"]) >= 0.02,
        10, lambda r: (float(r["edge"]), float(r["probabilidad_calibrada"])), 3, singles | combos,
    )
    return singles, combos, value

def main():
    series = sports_series(("KXNFL",))
    if not series:
        raise RuntimeError("No se encontró ninguna serie NFL KXNFL en el catálogo público de Kalshi.")
    # Skip player props and season futures: this model currently estimates game/team totals only.
    relevant = [item for item in series if any(token in str(item.get("ticker", "")).upper()
                for token in ("GAME", "SPREAD", "TOTAL", "TEAMTOTAL"))]
    if relevant:
        series = relevant
    raw_events = events_for_series([x["ticker"] for x in series], statuses=("open",))
    events = merge_game_events(raw_events)
    stats = _team_stats()
    excluded_ids = reconcile_exclusions(stats)
    print(f"NFL: equipos con estadísticas recientes={len(stats)}; capturas históricas excluidas={len(excluded_ids)}")
    print("NFL: muestra de eventos:", [(e.get("title"), e.get("start_time"), e.get("strike_date"), e.get("start_date"), len(e.get("markets") or [])) for e in events[:3]])
    if events and events[0].get("markets"):
        sample_market = events[0]["markets"][0]
        print("NFL: claves evento/mercado:", sorted(events[0].keys()), sorted(sample_market.keys()))
        print("NFL: campos de hora del mercado:", {k: sample_market.get(k) for k in ("close_time", "expiration_time", "expected_expiration_time", "open_time", "latest_expiration_time")})
    if events:
        fixture = str(events[0].get("title") or "").split(":", 1)[0].strip()
        teams = re.split(r"\s+(?:vs\.?|versus|@|at)\s+", fixture, maxsplit=1, flags=re.I)
        if len(teams) == 2:
            print(f"NFL: proyección de diagnóstico {teams[0]} vs {teams[1]}: {projected_score(teams[0], teams[1], stats)}")
    candidates = build_candidates(events, stats)
    print(f"NFL: partidos con proyección y mercados reconocidos={len(set(row['match_id'] for row in candidates))}")
    # Keep snapshots immutable; use market + side + event identity to avoid duplicates.
    existing = set()
    if PRED_FILE.exists():
        import csv
        with PRED_FILE.open(newline="", encoding="utf-8-sig") as f:
            existing = {r.get("prediction_id","") for r in csv.DictReader(f)}
    captured = datetime.now(timezone.utc).isoformat()
    rows = []
    for row in candidates:
        # Calibrate YES and NO independently; they are complementary contracts.
        category = f'{row["categoria"]} · {row["lado"]}'
        row["categoria"] = category
        calibrated, note = calibrate_probability(
            row["probabilidad_modelo"], category, PRED_FILE, RESULT_FILE,
            exclude_prediction_ids=excluded_ids,
        )
        row["probabilidad_calibrada"] = calibrated
        row["calibracion"] = note
        price = row["precio_captura"]
        row["edge"] = calibrated-price if price is not None else None
        row["bloque"] = ""
        row["capturado_en"] = captured
        base_id = f'{row["event_ticker"]}|{row["ticker"]}|{row["lado"]}'
        row["prediction_id"] = base_id + "|validado_prepartido" if base_id in excluded_ids else base_id
        if row["prediction_id"] not in existing:
            rows.append(row)
    single_ids, high_ids, value_ids = set(), set(), set()
    by_match = {}
    for row in candidates:
        by_match.setdefault(row["match_id"], []).append(row)
    for game_rows in by_match.values():
        singles, combos, value = _rank_nfl_game(game_rows)
        single_ids.update(singles)
        high_ids.update(combos)
        value_ids.update(value)
    for row in candidates:
        row["bloque"] = ("SENCILLAS" if row["prediction_id"] in single_ids else
                         "ALTA_PROBABILIDAD" if row["prediction_id"] in high_ids else
                         "A_TENER_EN_CUENTA" if row["prediction_id"] in value_ids else "DETALLE")
    # Current recommendations are recalculated per match; the historical ledger remains append-only.
    from csv import DictWriter
    current_fields = ["event_ticker","match_id","prediction_id","ticker","local","visita","fecha","categoria","mercado","lado","bloque","probabilidad_modelo","probabilidad_calibrada","precio_captura","edge","calibracion","marcador_proyectado"]
    with CURRENT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=current_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    append_immutable_predictions(PRED_FILE, PRED_COLUMNS, rows, probability_field="probabilidad_modelo")
    print(f"NFL: eventos brutos={len(raw_events)}, partidos agrupados={len(events)}")
    print(f"NFL: eventos abiertos={len(events)}, candidatos modelados={len(candidates)}, capturas nuevas={len(rows)}")

if __name__ == "__main__":
    main()
