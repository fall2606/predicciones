"""NFL forecasts for real Kalshi contracts; deliberately conservative on unsupported markets."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

from prediction_core import (
    KALSHI_API_BASE, append_immutable_predictions, valid_probability,
    sports_series, events_for_series, rank_market_candidates,
    rank_single_candidates, single_candidate_rejection_reasons,
)

ROOT = Path(__file__).resolve().parent
PRED_FILE = ROOT / "predicciones_nfl.csv"
RESULT_FILE = ROOT / "resultados_nfl.csv"
CATALOG_REPORT = ROOT / "nfl_catalogo_actual.csv"
REJECTED_REPORT = ROOT / "nfl_candidatos_descartados.csv"
TZ_ET = ZoneInfo("America/New_York")
TIMEOUT = int(os.getenv("NFL_HTTP_TIMEOUT", "12"))
NFLVERSE_GAMES_URL = os.getenv(
    "NFLVERSE_GAMES_URL",
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
)
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
API_KALSHI = KALSHI_API_BASE
SCHEMA_VERSION = "1"
PRED_COLUMNS = [
    "schema_version", "prediction_id", "event_ticker", "ticker", "match_id",
    "fecha", "local", "visita", "categoria", "mercado", "line", "lado",
    "seleccion", "bloque", "probabilidad_original", "probabilidad_calibrada",
    "version_calibracion", "probabilidad_modelo", "precio_captura",
    "probabilidad_implicita", "edge", "estado_modelo", "reglas",
    "capturado_en", "hora_inicio", "disponibilidad",
]
RESULT_COLUMNS = [
    "schema_version", "prediction_id", "estado", "resultado_kalshi",
    "marcador", "fuente", "motivo", "resuelto_en",
]
TEAM_ALIASES = {
    "ARI": ("arizona cardinals", "cardinals"), "ATL": ("atlanta falcons", "falcons"),
    "BAL": ("baltimore ravens", "ravens"), "BUF": ("buffalo bills", "bills"),
    "CAR": ("carolina panthers", "panthers"), "CHI": ("chicago bears", "bears"),
    "CIN": ("cincinnati bengals", "bengals"), "CLE": ("cleveland browns", "browns"),
    "DAL": ("dallas cowboys", "cowboys"), "DEN": ("denver broncos", "broncos"),
    "DET": ("detroit lions", "lions"), "GB": ("green bay packers", "packers"),
    "HOU": ("houston texans", "texans"), "IND": ("indianapolis colts", "colts"),
    "JAX": ("jacksonville jaguars", "jaguars"), "KC": ("kansas city chiefs", "chiefs"),
    "LAC": ("los angeles chargers", "chargers"), "LAR": ("los angeles rams", "rams"),
    "LV": ("las vegas raiders", "raiders"), "MIA": ("miami dolphins", "dolphins"),
    "MIN": ("minnesota vikings", "vikings"), "NE": ("new england patriots", "patriots"),
    "NO": ("new orleans saints", "saints"), "NYG": ("new york giants", "giants"),
    "NYJ": ("new york jets", "jets"), "PHI": ("philadelphia eagles", "eagles"),
    "PIT": ("pittsburgh steelers", "steelers"), "SEA": ("seattle seahawks", "seahawks"),
    "SF": ("san francisco 49ers", "49ers"), "TB": ("tampa bay buccaneers", "buccaneers"),
    "TEN": ("tennessee titans", "titans"), "WAS": ("washington commanders", "commanders"),
}
TEAM_NICKNAMES = {abbr: aliases[-1] for abbr, aliases in TEAM_ALIASES.items()}
TERMINAL = {"WIN", "LOSS", "ANULADA"}


def normalizar(value):
    raw = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", " ", raw).strip()


def parse_dt(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get(url, params=None):
    response = requests.get(url, params=params or {}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def discover_nfl_series():
    """Discover NFL series from Kalshi's live Sports series list; no ticker is assumed."""
    rows = sports_series()
    found = []
    for item in rows:
        ticker = str(item.get("ticker") or "").upper()
        title = normalizar(item.get("title") or item.get("name") or "")
        # Exclude college football and other football competitions.
        is_nfl = ("NFL" in ticker or "NATIONAL FOOTBALL LEAGUE" in title or
                  ("PRO FOOTBALL" in title and "COLLEGE" not in title))
        if is_nfl and not any(x in ticker for x in ("CFB", "NCAA", "COLLEGE")):
            found.append(item)
    return found


def load_kalshi_catalog():
    series = discover_nfl_series()
    if not series:
        raise RuntimeError("La API pública de Kalshi no devolvió series identificables como NFL.")
    events = events_for_series([str(s["ticker"]) for s in series], statuses=("open", "initialized"))
    # Keep event and nested market data exactly as returned by Kalshi.
    seen = set()
    catalog = []
    for event in events:
        key = str(event.get("event_ticker") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        markets = event.get("markets") or []
        catalog.append({**event, "markets": markets})
    return series, catalog


def _nflverse_games():
    df = pd.read_csv(NFLVERSE_GAMES_URL, low_memory=False)
    required = {"season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(df.columns):
        raise ValueError("El histórico nflverse no contiene las columnas mínimas para modelar la NFL.")
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"]).copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce", utc=True)
    df = df.dropna(subset=["gameday"])
    df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]
    return df.sort_values("gameday")


def build_team_model(games, as_of=None):
    """Own NFL model: recency-weighted team points, Elo, home field and NFL margin PMF."""
    cutoff = parse_dt(as_of) if as_of else datetime.now(timezone.utc)
    data = games.copy()
    data["gameday"] = pd.to_datetime(data["gameday"], utc=True, errors="coerce")
    data = data[data["gameday"] < pd.Timestamp(cutoff)]
    if data.empty:
        raise ValueError("No hay partidos NFL anteriores a la fecha de captura para entrenar el modelo.")
    data = data.tail(1800).copy()
    elo = {team: 1500.0 for team in TEAM_ALIASES}
    points_for = {team: [] for team in TEAM_ALIASES}
    points_against = {team: [] for team in TEAM_ALIASES}
    # Historical games are replayed chronologically; the current game is never used.
    for row in data.to_dict("records"):
        home, away = str(row["home_team"]).upper(), str(row["away_team"]).upper()
        if home not in elo or away not in elo:
            continue
        hs, aws = float(row["home_score"]), float(row["away_score"])
        h_elo, a_elo = elo[home], elo[away]
        expected_home = 1 / (1 + 10 ** (-(h_elo + 55 - a_elo) / 400))
        actual_home = 1.0 if hs > aws else (0.5 if hs == aws else 0.0)
        margin = abs(hs - aws)
        k = 20 * math.log1p(margin) / math.log(11) if margin else 12
        change = k * (actual_home - expected_home)
        elo[home] += change
        elo[away] -= change
        points_for[home].append((hs, row["gameday"]))
        points_against[home].append((aws, row["gameday"]))
        points_for[away].append((aws, row["gameday"]))
        points_against[away].append((hs, row["gameday"]))
    def recent_average(values, fallback):
        if not values:
            return fallback
        values = values[-12:]
        weights = np.exp(np.linspace(-1.2, 0, len(values)))
        result = float(np.average([x[0] for x in values], weights=weights))
        return result if math.isfinite(result) and result >= 0 else fallback
    league_points = float(np.mean(
        list(data["home_score"].tail(250)) + list(data["away_score"].tail(250))
    ))
    if not math.isfinite(league_points) or league_points <= 0:
        league_points = 21.5
    teams = {}
    for team in TEAM_ALIASES:
        teams[team] = {
            "elo": float(elo[team]),
            "pf": recent_average(points_for[team], league_points),
            "pa": recent_average(points_against[team], league_points),
            "sample": len(points_for[team]),
        }
    return {"teams": teams, "league_points": league_points, "trained_games": len(data)}


def _normal_pmf(mean, sd, low, high, *, key_numbers=()):
    values = np.arange(low, high + 1, dtype=int)
    sd = max(float(sd), 1.0)
    probs = norm.cdf((values + 0.5 - mean) / sd) - norm.cdf((values - 0.5 - mean) / sd)
    # Mild, empirically motivated extra mass at NFL key margins (3, 7, and combinations).
    for number in key_numbers:
        if low <= number <= high:
            probs[number - low] *= 1.12 if abs(number) in (3, 7) else 1.04
    total = float(probs.sum())
    if not math.isfinite(total) or total <= 0:
        return values, np.full(len(values), 1.0 / len(values))
    return values, probs / total


def predict_game(model, home, away):
    h = model["teams"].get(home)
    a = model["teams"].get(away)
    if not h or not a:
        return None
    margin_mean = (h["elo"] - a["elo"]) / 25.0 + 1.7
    home_mean = max(0.0, (h["pf"] + a["pa"]) / 2 + 1.0)
    away_mean = max(0.0, (a["pf"] + h["pa"]) / 2 - 1.0)
    total_mean = home_mean + away_mean
    margin_values, margin_probs = _normal_pmf(margin_mean, 13.5, -45, 45, key_numbers=(-14, -10, -7, -6, -3, 0, 3, 6, 7, 10, 14))
    total_values, total_probs = _normal_pmf(total_mean, 13.0, 0, 100, key_numbers=(3, 7, 10, 14, 17, 20, 21, 24, 27, 28, 31, 34, 35, 38, 41, 42, 45, 48, 49, 52, 55))
    return {
        "margin_mean": margin_mean, "total_mean": total_mean,
        "home_points_mean": home_mean, "away_points_mean": away_mean,
        "home_win": float(margin_probs[margin_values > 0].sum()),
        "away_win": float(margin_probs[margin_values < 0].sum()),
        "margin_values": margin_values, "margin_probs": margin_probs,
        "total_values": total_values, "total_probs": total_probs,
    }


def _price(market, side):
    keys = (f"{side.lower()}_ask_dollars", f"{side.lower()}_bid_dollars")
    for key in keys:
        try:
            value = float(market.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and 0 < value < 1:
            return value
    # Older API payloads may only expose integer-cent quotes.
    for key in (f"{side.lower()}_ask", f"{side.lower()}_bid"):
        try:
            value = float(market.get(key)) / 100
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and 0 < value < 1:
            return value
    return None


def _event_teams(event):
    title = normalizar(event.get("title") or event.get("sub_title") or "")
    found = []
    for abbr, aliases in TEAM_ALIASES.items():
        if any(re.search(r"\b" + re.escape(normalizar(alias)) + r"\b", title) for alias in aliases):
            found.append(abbr)
    # Ticker matching is only a secondary hint; never accept an ambiguous event.
    ticker = str(event.get("event_ticker") or "").upper()
    if len(found) < 2:
        for abbr in TEAM_ALIASES:
            if re.search(r"\b" + re.escape(abbr) + r"\b", ticker):
                found.append(abbr)
    found = list(dict.fromkeys(found))
    return found if len(found) == 2 else []


def _kickoff(event):
    for key in ("start_time", "expected_expiration_time", "strike_date", "close_time"):
        dt = parse_dt(event.get(key))
        if dt:
            return dt
    # Kalshi's event title/ticker time is not guessed if the API omitted a timestamp.
    return None


def _market_type(market, series_ticker):
    text = normalizar(" ".join(str(market.get(k) or "") for k in ("title", "subtitle", "yes_sub_title", "no_sub_title")))
    series = str(series_ticker or market.get("series_ticker") or "").upper()
    if any(token in series for token in ("PLAYER", "PASSYDS", "RUSHYDS", "RECYDS", "RECEPTIONS", "TOUCHDOWN", "TD")):
        return "unsupported_player_prop"
    if any(token in series for token in ("1Q", "2Q", "3Q", "4Q", "1H", "2H", "HALF", "QUARTER")):
        return "unsupported_period"
    if "TEAMTOTAL" in series or "TEAM TOTAL" in text or "TEAM POINTS" in text:
        return "team_total"
    if "SPREAD" in series or "HANDICAP" in text or "SPREAD" in text:
        return "spread"
    if "TOTAL" in series or re.search(r"\b(over|under|total points|combined points)\b", text):
        return "total"
    if "GAME" in series or any(token in text for token in ("win the game", "to win", "winner")):
        return "winner"
    return "unsupported_other"


def _line_from_market(market):
    text = " ".join(str(market.get(k) or "") for k in ("yes_sub_title", "title", "subtitle", "ticker"))
    patterns = [
        r"(?i)\b(?:over|under|at least|at most|more than|less than)\s*\$?(-?\d+(?:\.\d+)?)",
        r"(?i)(-?\d+(?:\.\d+)?)\s*(?:\+ points|or more|or higher|points or more)",
        r"(?<![A-Za-z])(-?\d+\.\d+)(?![A-Za-z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                value = float(match.group(1))
                if math.isfinite(value):
                    return value
            except ValueError:
                pass
    return None


def _yes_team(market, home, away):
    text = normalizar(" ".join(str(market.get(k) or "") for k in ("yes_sub_title", "title", "subtitle")))
    matched = []
    for team in (home, away):
        if any(re.search(r"\b" + re.escape(normalizar(alias)) + r"\b", text) for alias in TEAM_ALIASES[team]):
            matched.append(team)
    return matched[0] if len(set(matched)) == 1 else None


def model_market(market, series_ticker, game, prediction):
    """Return P(YES) only for exact, supported full-game market semantics."""
    kind = _market_type(market, series_ticker)
    title = str(market.get("yes_sub_title") or market.get("title") or "")
    line = _line_from_market(market)
    if kind == "winner":
        team = _yes_team(market, game["home"], game["away"])
        if not team:
            # Winner contracts often encode the YES team in the real market ticker.
            ticker = str(market.get("ticker") or "").upper()
            candidates = [t for t in (game["home"], game["away"]) if ticker.endswith("-" + t)]
            team = candidates[0] if len(candidates) == 1 else None
        if not team:
            return None, "ganador sin equipo SÍ identificable de forma inequívoca"
        p = prediction["home_win"] if team == game["home"] else prediction["away_win"]
        return valid_probability(p), "Elo NFL + puntos recientes; temporada regular/prórroga pendiente de regla Kalshi"
    if kind == "spread":
        team = _yes_team(market, game["home"], game["away"])
        if not team or line is None:
            return None, "hándicap sin equipo SÍ y línea exacta verificables"
        values, probs = prediction["margin_values"], prediction["margin_probs"]
        # Kalshi lines are interpreted literally; integer 'N+' means margin >= N.
        if "+" in title or re.search(r"\b(?:at least|or more)\b", normalizar(title)):
            threshold = math.ceil(line)
            p = probs[values >= threshold].sum() if team == game["home"] else probs[values <= -threshold].sum()
        else:
            p = probs[values > line].sum() if team == game["home"] else probs[values < -line].sum()
        return valid_probability(float(p)), "Distribución discreta de margen NFL; prórroga según reglas del contrato"
    if kind == "total":
        if line is None:
            return None, "total sin umbral explícito en título/subtítulo de Kalshi"
        text = normalizar(title)
        values, probs = prediction["total_values"], prediction["total_probs"]
        if re.search(r"\bunder\b|\bless than\b|\bmenos de\b", text):
            p = probs[values < line].sum()
        elif re.search(r"\bover\b|\bmore than\b|\bat least\b|\bor more\b|\+\s*$", text):
            p = probs[values >= math.ceil(line)].sum()
        else:
            # Do not guess whether an integer threshold means >N or >=N.
            return None, "semántica del umbral total ambigua; no se genera"
        return valid_probability(float(p)), "Distribución discreta del total NFL; prórroga según reglas del contrato"
    if kind == "team_total":
        team = _yes_team(market, game["home"], game["away"])
        if not team or line is None:
            return None, "total por equipo sin equipo/línea inequívocos"
        mean = prediction["home_points_mean"] if team == game["home"] else prediction["away_points_mean"]
        values, probs = _normal_pmf(mean, 9.5, 0, 70)
        text = normalizar(title)
        if "under" in text or "less than" in text:
            p = probs[values < line].sum()
        elif "over" in text or "or more" in text or "+" in title:
            p = probs[values >= math.ceil(line)].sum()
        else:
            return None, "semántica del total por equipo ambigua"
        return valid_probability(float(p)), "Distribución de puntos del equipo; prórroga según reglas del contrato"
    if kind == "unsupported_player_prop":
        return None, "prop de jugador no se genera: aún no hay modelo/participación individual validados"
    if kind == "unsupported_period":
        return None, "mercado por mitad/cuarto no se genera sin histórico oficial por período validado"
    return None, "tipo de mercado no soportado por el modelo NFL"


def _canonical_team_code(value):
    code = str(value or "").upper().strip()
    return {"LA": "LAR", "JAC": "JAX", "JAX": "JAX", "OAK": "LV", "WFT": "WAS", "WSH": "WAS"}.get(code, code)


def load_upcoming_games(days=21, now=None):
    """Load the public ESPN NFL schedule; team orientation comes from homeAway, not title order."""
    start = (parse_dt(now) if now else datetime.now(timezone.utc)).date()
    games = []
    for offset in range(max(1, int(days))):
        day = start + timedelta(days=offset)
        payload = _get(ESPN_SCOREBOARD_URL, {"dates": day.strftime("%Y%m%d"), "limit": 100})
        for event in payload.get("events") or []:
            competitions = event.get("competitions") or []
            if not competitions:
                continue
            comp = competitions[0]
            teams = {}
            for competitor in comp.get("competitors") or []:
                side = str(competitor.get("homeAway") or "").lower()
                code = _canonical_team_code((competitor.get("team") or {}).get("abbreviation"))
                if side in {"home", "away"} and code in TEAM_ALIASES:
                    teams[side] = code
            kickoff = parse_dt(comp.get("date") or event.get("date"))
            if kickoff and "home" in teams and "away" in teams:
                games.append({
                    "home": teams["home"], "away": teams["away"], "date": kickoff.isoformat(),
                    "match_id": f"NFL:{kickoff.date().isoformat()}:{teams['away']}@{teams['home']}",
                    "status": (event.get("status") or {}).get("type", {}).get("name", ""),
                    "espn_id": str(event.get("id") or ""),
                })
    return games


def _match_catalog_event(event, event_teams, upcoming_games, event_kickoff):
    if len(event_teams) != 2:
        return None
    matches = []
    for game in upcoming_games:
        if set(event_teams) != {game["home"], game["away"]}:
            continue
        game_dt = parse_dt(game.get("date"))
        if not game_dt or not event_kickoff or abs((game_dt - event_kickoff).total_seconds()) > 36 * 3600:
            continue
        matches.append(game)
    if len(matches) != 1:
        return None
    return matches[0]


def _match_id(kickoff, home, away):
    return f"NFL:{kickoff.date().isoformat()}:{away}@{home}"


def generate_candidates(events, model, now=None, upcoming_games=None):
    now = parse_dt(now) if now else datetime.now(timezone.utc)
    rows, rejected, catalog = [], [], []
    for event in events:
        kickoff = _kickoff(event)
        teams = _event_teams(event)
        if not kickoff or len(teams) != 2 or kickoff <= now:
            continue
        # ESPN/nflverse abbreviations are the canonical order only when event metadata states it.
        # Kalshi event titles may list away first; market team semantics remain explicit.
        title = normalizar(event.get("title") or event.get("sub_title") or "")
        home = next((t for t in teams if any(normalizar(a) in title for a in TEAM_ALIASES[t])), None)
        away = next((t for t in teams if t != home), None)
        if not home or not away:
            continue
        game = {"home": home, "away": away, "date": kickoff.isoformat(), "event_ticker": event.get("event_ticker", "")}
        prediction = predict_game(model, home, away)
        if not prediction:
            continue
        match_id = _match_id(kickoff, home, away)
        for market in event.get("markets") or []:
            ticker = str(market.get("ticker") or "").strip()
            if not ticker:
                continue
            series = str(market.get("series_ticker") or event.get("series_ticker") or "")
            market = {**market, "series_ticker": series}
            p_yes, reason = model_market(market, series, game, prediction)
            catalog.append({
                "event_ticker": event.get("event_ticker", ""), "ticker": ticker,
                "series_ticker": series, "event_title": event.get("title") or event.get("sub_title") or "",
                "market_title": market.get("yes_sub_title") or market.get("title") or "",
                "market_status": market.get("status") or "", "yes_bid_dollars": market.get("yes_bid_dollars"),
                "yes_ask_dollars": market.get("yes_ask_dollars"), "no_bid_dollars": market.get("no_bid_dollars"),
                "no_ask_dollars": market.get("no_ask_dollars"), "modelable": "SI" if valid_probability(p_yes) is not None else "NO",
                "motivo": "" if valid_probability(p_yes) is not None else reason,
            })
            if valid_probability(p_yes) is None:
                rejected.append({"match_id": match_id, "local": home, "visita": away, "event_ticker": event.get("event_ticker", ""), "ticker": ticker, "categoria": _market_type(market, series), "motivo": reason})
                continue
            for side, p in (("YES", p_yes), ("NO", 1 - p_yes)):
                price = _price(market, side)
                edge = p - price if price is not None else None
                captured = now.isoformat()
                pid = hashlib.sha256(f"{ticker}|{side}|{captured}".encode()).hexdigest()[:24]
                rows.append({
                    "schema_version": SCHEMA_VERSION, "prediction_id": pid,
                    "event_ticker": event.get("event_ticker", ""), "ticker": ticker,
                    "match_id": match_id, "fecha": kickoff.isoformat(), "hora_inicio": kickoff.isoformat(),
                    "local": home, "visita": away, "categoria": _market_type(market, series),
                    "mercado": market.get("yes_sub_title") or market.get("title") or ticker,
                    "line": _line_from_market(market) if _line_from_market(market) is not None else "",
                    "lado": side, "seleccion": side, "bloque": "",
                    "probabilidad_original": f"{p:.8f}", "probabilidad_calibrada": "",
                    "version_calibracion": "calibración insuficiente",
                    "probabilidad_modelo": f"{p:.8f}",
                    "precio_captura": f"{price:.6f}" if price is not None else "",
                    "probabilidad_implicita": f"{price:.6f}" if price is not None else "",
                    "edge": f"{edge:.6f}" if edge is not None else "",
                    "estado_modelo": "NFL Elo + forma reciente; sin calibración por falta de muestra auditada",
                    "reglas": market.get("rules_primary") or market.get("rules_secondary") or "",
                    "capturado_en": captured,
                    "disponibilidad": "Disponible en Kalshi" if price is not None else "Sin precio en Kalshi",
                })
    return rows, rejected, catalog


def save_capture(rows, path=PRED_FILE):
    """Shared append-only writer; rejects any prediction captured at/after kickoff."""
    return append_immutable_predictions(
        path, PRED_COLUMNS, rows, probability_field="probabilidad_modelo",
        kickoff_field="hora_inicio", capture_field="capturado_en",
    )


def _write_report(path, columns, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def capture():
    """Bounded live capture; called by the shared scheduled pipeline."""
    started = time.monotonic()
    games = _nflverse_games()
    model = build_team_model(games)
    series, events = load_kalshi_catalog()
    upcoming = load_upcoming_games()
    rows, rejected, catalog = generate_candidates(events, model, upcoming_games=upcoming)
    # Archive full observed catalog and rejection reasons for auditability.
    _write_report(CATALOG_REPORT,
        ["event_ticker", "ticker", "series_ticker", "event_title", "market_title", "market_status",
         "yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars", "modelable", "motivo"], catalog)
    _write_report(REJECTED_REPORT,
        ["match_id", "local", "visita", "event_ticker", "ticker", "categoria", "motivo"], rejected)
    saved = save_capture(rows)
    print(f"NFL catálogo: series={len(series)} eventos={len(events)} mercados={len(catalog)} modelables={sum(x['modelable']=='SI' for x in catalog)} descartados={len(rejected)}")
    print(f"NFL captura: aceptadas={saved['accepted']} rechazadas={saved['rejected']} histórico_modelo={model['trained_games']} duración_s={time.monotonic()-started:.1f}")
    return {"series": series, "events": events, "catalog": catalog, "saved": saved, "rejected": rejected}


def settle():
    """Resolve all stored predictions from Kalshi first; conservative ESPN fallback for full-game markets."""
    import json
    import requests
    from prediction_core import kalshi_get
    from sports_results import settle_from_score
    predictions = []
    if PRED_FILE.exists():
        with PRED_FILE.open(newline="", encoding="utf-8-sig") as handle:
            predictions = list(csv.DictReader(handle))
    results = []
    if RESULT_FILE.exists():
        with RESULT_FILE.open(newline="", encoding="utf-8-sig") as handle:
            results = list(csv.DictReader(handle))
    latest = {}
    for row in results:
        pid = str(row.get("prediction_id") or "")
        if pid and (pid not in latest or str(row.get("resuelto_en", "")) > str(latest[pid].get("resuelto_en", ""))):
            latest[pid] = row
    new_results = []
    for row in predictions:
        pid = str(row.get("prediction_id") or "")
        if not pid or str(latest.get(pid, {}).get("estado", "")).upper() in TERMINAL:
            continue
        ticker = str(row.get("ticker") or "")
        try:
            market_data = kalshi_get(f"markets/{ticker}").get("market", {})
        except (requests.RequestException, ValueError, KeyError):
            market_data = {}
        kalshi_result = str(market_data.get("result") or "").lower()
        if kalshi_result in {"void", "canceled", "cancelled"}:
            state, source, reason = "ANULADA", "Kalshi", "contrato anulado/cancelado por Kalshi"
        elif kalshi_result in {"yes", "no"}:
            side = str(row.get("lado") or row.get("seleccion") or "").upper()
            if side not in {"YES", "NO"}:
                continue
            state = "WIN" if side.lower() == kalshi_result else "LOSS"
            source, reason = "Kalshi", "resultado oficial del ticker"
        else:
            # Do not guess from final score if exact contract semantics were not stored.
            fallback = settle_from_score({
                "fecha": row.get("fecha"), "local": row.get("local"), "visita": row.get("visita"),
                "ticker": ticker, "mercado": row.get("mercado"), "line": row.get("line"),
                "lado": row.get("lado"), "seleccion": row.get("seleccion"),
                "deporte": "NFL", "competition": "NFL",
            })
            if not fallback or fallback.get("state") not in TERMINAL:
                continue
            state, source, reason = fallback["state"], "fuente independiente", fallback.get("reason", "")
        new_results.append({
            "schema_version": SCHEMA_VERSION, "prediction_id": pid, "estado": state,
            "resultado_kalshi": kalshi_result, "marcador": "", "fuente": source,
            "motivo": reason, "resuelto_en": datetime.now(timezone.utc).isoformat(),
        })
    if new_results:
        _write_report(RESULT_FILE, RESULT_COLUMNS, results + new_results)
    print(f"NFL liquidación: nuevas={len(new_results)} pendientes_previas={sum(1 for p in predictions if p.get('prediction_id') not in latest)}")
    return new_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "settle", "catalog-smoke"))
    args = parser.parse_args()
    if args.action == "capture":
        capture()
    elif args.action == "settle":
        settle()
    else:
        series, events = load_kalshi_catalog()
        print(json.dumps({
            "series": [{"ticker": x.get("ticker"), "title": x.get("title")} for x in series],
            "example_event": next(({
                "event_ticker": e.get("event_ticker"), "title": e.get("title"),
                "markets": [{"ticker": m.get("ticker"), "title": m.get("title") or m.get("yes_sub_title"),
                             "yes_ask_dollars": m.get("yes_ask_dollars"), "no_ask_dollars": m.get("no_ask_dollars")}
                            for m in (e.get("markets") or [])[:12]}
            } for e in events if e.get("markets")), None),
        }, indent=2))
