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
    sports_series, iter_kalshi_pages, events_for_series, rank_market_candidates,
    rank_single_candidates, single_candidate_rejection_reasons,
)

ROOT = Path(__file__).resolve().parent
PRED_FILE = ROOT / "predicciones_nfl.csv"
RESULT_FILE = ROOT / "resultados_nfl.csv"
CATALOG_REPORT = ROOT / "nfl_catalogo_actual.csv"
REJECTED_REPORT = ROOT / "nfl_candidatos_descartados.csv"
COVERAGE_REPORT = ROOT / "nfl_cobertura_por_partido.csv"
COVERAGE_COLUMNS = ["match_id", "fecha", "local", "visita", "contratos_modelables", "sencillas_validas", "alta_probabilidad_validas", "a_tener_en_cuenta_validas", "motivo_sencillas", "motivo_alta_probabilidad", "motivo_a_tener_en_cuenta"]
TZ_ET = ZoneInfo("America/New_York")
TIMEOUT = int(os.getenv("NFL_HTTP_TIMEOUT", "12"))
NFLVERSE_GAMES_URL = os.getenv(
    "NFLVERSE_GAMES_URL",
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
)
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
API_KALSHI = KALSHI_API_BASE
ESPN_SCOREBOARD_CACHE = {}
SCHEMA_VERSION = "1"
PRED_COLUMNS = [
    "schema_version", "prediction_id", "event_ticker", "ticker", "match_id",
    "fecha", "local", "visita", "categoria", "mercado", "line", "lado", "equipo_yes", "tipo_salida", "descripcion",
    "seleccion", "bloque", "probabilidad_original", "probabilidad_calibrada",
    "version_calibracion", "probabilidad_modelo", "precio_captura",
    "probabilidad_implicita", "edge", "revision", "estado_modelo", "reglas",
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
    """Discover game-related NFL series from Kalshi's public API, not assumed ticker lists."""
    try:
        # Kalshi's public API supports tags=Football; use it to avoid paging through every sport.
        rows = iter_kalshi_pages(
            "series", {"limit": 200, "category": "Sports", "tags": "Football"},
            "series", 10,
        )
    except (requests.RequestException, ValueError, KeyError):
        rows = sports_series()
    found = []
    include_terms = (
        "GAME", "SPREAD", "TOTAL", "TEAMTOTAL", "PLAYER", "PASSYDS", "RUSHYDS",
        "RECYDS", "RECEPTIONS", "TOUCHDOWN", "1H", "2H", "1Q", "2Q", "3Q", "4Q",
        "HALF", "QUARTER",
    )
    exclude_terms = ("SUPERBOWL", "CHAMPION", "MVP", "PLAYOFFS", "SEASON", "COACH", "AWARD", "DRAFT", "TRADE", "CAREER", "DIVISIONWINS", "LASTTOWIN", "TDPROT")
    for item in rows:
        ticker = str(item.get("ticker") or "").upper()
        title = normalizar(item.get("title") or item.get("name") or "")
        is_nfl = ("NFL" in ticker or "NATIONAL FOOTBALL LEAGUE" in title or
                  ("PRO FOOTBALL" in title and "COLLEGE" not in title))
        game_related = any(token in ticker or token.casefold() in title for token in include_terms)
        if (is_nfl and game_related
                and not any(x in ticker for x in ("CFB", "NCAA", "COLLEGE"))
                and not any(x in ticker for x in exclude_terms)):
            found.append(item)
    # Stable unique list avoids duplicate API requests if the series endpoint repeats a row.
    unique = {str(row.get("ticker") or ""): row for row in found if row.get("ticker")}
    return list(unique.values())

def load_kalshi_catalog():
    series = discover_nfl_series()
    if not series:
        raise RuntimeError("La API pública de Kalshi no devolvió series identificables como NFL.")
    events = []
    # Upcoming match markets are open; avoid the much slower initialized/futures archive.
    # Two pages per series keep the live capture bounded.
    for item in series:
        try:
            rows = iter_kalshi_pages(
                "events",
                {"limit": 200, "series_ticker": str(item["ticker"]), "status": "open",
                 "with_nested_markets": "true"},
                "events", 2,
            )
            events.extend(rows)
        except (requests.RequestException, ValueError, KeyError) as exc:
            print("AVISO catálogo NFL {}: {}".format(item.get("ticker"), exc))
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
    df["home_team"] = df["home_team"].map(_canonical_team_code)
    df["away_team"] = df["away_team"].map(_canonical_team_code)
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
        "home_win": float(margin_probs[margin_values > 0].sum() + 0.5 * margin_probs[margin_values == 0].sum()),
        "away_win": float(margin_probs[margin_values < 0].sum() + 0.5 * margin_probs[margin_values == 0].sum()),
        "margin_values": margin_values, "margin_probs": margin_probs,
        "total_values": total_values, "total_probs": total_probs,
    }


def _price(market, side):
    # Only asks represent the price to buy; a bid is not a buy price.
    for key in (f"{side.lower()}_ask_dollars", f"{side.lower()}_ask"):
        try:
            value = float(market.get(key))
            if key.endswith("_ask") and not key.endswith("_ask_dollars"):
                value /= 100
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and 0 < value < 1:
            return value
    return None

def _event_teams(event):
    title = normalizar(" ".join(str(event.get(k) or "") for k in ("title", "sub_title", "subtitle")))
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
    # Many Kalshi NFL game events encode only the ET calendar date in the event ticker.
    # Do not invent a kickoff time; use the official ESPN schedule to supply the exact time.
    return None


def _event_date(event):
    kickoff = _kickoff(event)
    if kickoff:
        return kickoff.astimezone(TZ_ET).date()
    ticker = str(event.get("event_ticker") or "").upper()
    match = re.search(r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(?:\D|$)", ticker)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%y%b%d").date()
    except ValueError:
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
        r"(?i)(?:win by|wins by|by|spread)\s*(-?\d+(?:\.\d+)?)\s*\+?",
        r"(?i)(-?\d+(?:\.\d+)?)\s*\+",
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

def _market_yes_team(market, home, away):
    team = _yes_team(market, home, away)
    if team:
        return team
    suffix = str(market.get("ticker") or "").upper().rsplit("-", 1)[-1]
    matches = [team for team in (home, away) if suffix == team or suffix.startswith(team)]
    return matches[0] if len(matches) == 1 else None


def _yes_outcome(market, kind):
    raw = " ".join(str(market.get(k) or "") for k in ("yes_sub_title", "title", "subtitle")).casefold()
    text = normalizar(raw)
    if kind in {"total", "team_total"}:
        if any(token in text for token in ("at most", "or fewer", "no more than")):
            return "under_le"
        if re.search(r"\bunder\b|\bless than\b|\bmenos de\b", text):
            return "under_lt"
        if any(token in text for token in ("at least", "or more", "or higher")) or re.search(r"\d\s*\+", raw):
            return "over_ge"
        if re.search(r"\bover\b|\bmore than\b", text):
            return "over_gt"
    if kind == "spread":
        if re.search(r"\b(?:win by|wins by|ganar por|ganan por|at least|or more)\b", text) or re.search(r"\d\s*\+", raw):
            return "win_by"
        return "spread_cover"
    return kind

def _team_label(team):
    return TEAM_ALIASES.get(team, (str(team),))[0].title()


def _description(market, kind, home, away, side, line):
    yes_team = _market_yes_team(market, home, away)
    yes_outcome = _yes_outcome(market, kind)
    if kind in {"winner", "spread", "team_total"} and not yes_team:
        return "", "", ""
    if kind == "winner":
        proposition = f"los {_team_label(yes_team)} ganan el partido"
        desc = f"SÍ: {proposition}" if side == "YES" else f"NO: los {_team_label(yes_team)} no ganan el partido"
    elif kind == "spread":
        amount = f"{abs(float(line)):g}" if line not in (None, "") else ""
        if yes_outcome == "win_by":
            proposition = f"los {_team_label(yes_team)} ganan por {amount} o más puntos"
            desc = f"SÍ: {proposition}" if side == "YES" else f"NO: los {_team_label(yes_team)} no ganan por {amount} o más puntos"
        else:
            signed = f"{float(line):+g}"
            desc = f"SÍ: los {_team_label(yes_team)} cubren el hándicap {signed}" if side == "YES" else f"NO: los {_team_label(yes_team)} no cubren el hándicap {signed}"
    elif kind == "total":
        amount = f"{float(line):g}" if line not in (None, "") else ""
        if yes_outcome.startswith("over"):
            desc = f"SÍ: el partido supera {amount} puntos en total" if side == "YES" else f"NO: el partido no supera {amount} puntos en total"
        elif yes_outcome.startswith("under"):
            desc = f"SÍ: el partido termina por debajo de {amount} puntos" if side == "YES" else f"NO: el partido no termina por debajo de {amount} puntos"
        else:
            return "", yes_outcome, yes_team or ""
    elif kind == "team_total":
        amount = f"{float(line):g}" if line not in (None, "") else ""
        if yes_outcome.startswith("over"):
            desc = f"SÍ: los {_team_label(yes_team)} superan {amount} puntos" if side == "YES" else f"NO: los {_team_label(yes_team)} no superan {amount} puntos"
        elif yes_outcome.startswith("under"):
            desc = f"SÍ: los {_team_label(yes_team)} se quedan por debajo de {amount} puntos" if side == "YES" else f"NO: los {_team_label(yes_team)} no se quedan por debajo de {amount} puntos"
        else:
            return "", yes_outcome, yes_team
    else:
        return "", yes_outcome, yes_team or ""
    return desc, yes_outcome, yes_team or ""

def model_market(market, series_ticker, game, prediction):
    """Return P(YES) only for exact, supported full-game market semantics."""
    kind = _market_type(market, series_ticker)
    title = str(market.get("yes_sub_title") or market.get("title") or "")
    line = _line_from_market(market)
    if kind == "winner":
        team = _market_yes_team(market, game["home"], game["away"])
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
        team = _market_yes_team(market, game["home"], game["away"])
        if not team or line is None:
            return None, "hándicap sin equipo SÍ y línea exacta verificables"
        text = normalizar(title)
        values, probs = prediction["margin_values"], prediction["margin_probs"]
        if re.search(r"\b(?:win by|wins by|ganar por|ganan por|at least|or more)\b", text) or "+" in title:
            threshold = abs(float(line))
            p = probs[values >= math.ceil(threshold)].sum() if team == game["home"] else probs[values <= -math.ceil(threshold)].sum()
        elif "cover" in text or "spread" in text or re.search(r"(?<!\w)[+-]\d+(?:\.\d+)?", title):
            # A negative line is a favorite laying points; a positive line is an underdog receiving points.
            signed_line = float(line)
            threshold = abs(signed_line)
            if signed_line < 0:
                p = probs[values > threshold].sum() if team == game["home"] else probs[values < -threshold].sum()
            else:
                p = probs[values > -threshold].sum() if team == game["home"] else probs[values < threshold].sum()
        else:
            return None, "semántica del hándicap ambigua; no se genera"
        return valid_probability(float(p)), "Distribución discreta del margen NFL; línea interpretada literalmente; prórroga según reglas Kalshi"

    if kind == "total":
        if line is None:
            return None, "total sin umbral explícito en título/subtítulo de Kalshi"
        text = normalizar(title)
        values, probs = prediction["total_values"], prediction["total_probs"]
        outcome = _yes_outcome(market, kind)
        if outcome == "under_le":
            p = probs[values <= math.floor(line)].sum()
        elif outcome == "under_lt":
            p = probs[values < line].sum()
        elif outcome == "over_ge":
            p = probs[values >= math.ceil(line)].sum()
        elif outcome == "over_gt":
            p = probs[values > line].sum()
        else:
            return None, "semántica del umbral total ambigua; no se genera"
        return valid_probability(float(p)), "Distribución discreta del total NFL; prórroga según reglas del contrato"
    if kind == "team_total":
        team = _market_yes_team(market, game["home"], game["away"])
        if not team or line is None:
            return None, "total por equipo sin equipo/línea inequívocos"
        mean = prediction["home_points_mean"] if team == game["home"] else prediction["away_points_mean"]
        values, probs = _normal_pmf(mean, 9.5, 0, 70)
        outcome = _yes_outcome(market, kind)
        if outcome == "under_le":
            p = probs[values <= math.floor(line)].sum()
        elif outcome == "under_lt":
            p = probs[values < line].sum()
        elif outcome == "over_ge":
            p = probs[values >= math.ceil(line)].sum()
        elif outcome == "over_gt":
            p = probs[values > line].sum()
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
                    "match_id": f"NFL:{kickoff.astimezone(TZ_ET).date().isoformat()}:{teams['away']}@{teams['home']}",
                    "status": (event.get("status") or {}).get("type", {}).get("name", ""),
                    "espn_id": str(event.get("id") or ""),
                })
    return games


def _match_catalog_event(event, event_teams, upcoming_games, event_kickoff):
    if len(event_teams) != 2:
        return None
    event_date = _event_date(event)
    matches = []
    for game in upcoming_games:
        if set(event_teams) != {game["home"], game["away"]}:
            continue
        game_dt = parse_dt(game.get("date"))
        if not game_dt:
            continue
        if event_kickoff:
            if abs((game_dt - event_kickoff).total_seconds()) > 36 * 3600:
                continue
        elif not event_date or game_dt.astimezone(TZ_ET).date() != event_date:
            continue
        matches.append(game)
    if len(matches) != 1:
        return None
    return matches[0]


def _match_id(kickoff, home, away):
    return f"NFL:{kickoff.astimezone(TZ_ET).date().isoformat()}:{away}@{home}"


def _edge_number(candidate):
    try:
        value = float(candidate.get("Ventaja (edge)", candidate.get("edge", "")))
        return value if math.isfinite(value) else -999.0
    except (TypeError, ValueError):
        return -999.0


def _market_family(row):
    kind = str(row.get("categoria") or "")
    if kind in {"winner", "spread"}:
        return "resultado"
    if kind == "team_total":
        return "total_equipo:" + str(row.get("equipo_yes") or "desconocido")
    if kind == "total":
        return "total_partido"
    return kind


def _reason_for_empty(block, candidates):
    if not candidates:
        return "no hay contratos con modelo NFL verificable para este partido"
    reasons = single_candidate_rejection_reasons(candidates)
    if block == "SENCILLAS":
        return "ninguna línea cumple precio máximo, ventaja mínima y diversificación: " + ", ".join(f"{k}={v}" for k, v in reasons.items())
    if block == "ALTA_PROBABILIDAD":
        return "ninguna línea cumple probabilidad mínima y precio máximo de 90¢ sin correlación"
    return "ninguna línea con precio y ventaja calculables supera los filtros del bloque"


def generate_candidates(events, model, now=None, upcoming_games=None):
    now = parse_dt(now) if now else datetime.now(timezone.utc)
    raw_by_match, rejected, catalog, match_info = {}, [], [], {}
    upcoming_games = upcoming_games if upcoming_games is not None else load_upcoming_games(now=now)
    for event in events:
        event_kickoff = _kickoff(event)
        teams = _event_teams(event)
        if len(teams) != 2:
            continue
        fixture = _match_catalog_event(event, teams, upcoming_games, event_kickoff)
        if not fixture:
            continue
        home, away = fixture["home"], fixture["away"]
        kickoff = parse_dt(fixture["date"])
        if not kickoff or kickoff <= now:
            continue
        game = {"home": home, "away": away, "date": kickoff.isoformat(), "event_ticker": event.get("event_ticker", "")}
        prediction = predict_game(model, home, away)
        if not prediction:
            continue
        match_id = _match_id(kickoff, home, away)
        match_info[match_id] = {"fecha": kickoff.isoformat(), "local": home, "visita": away}
        raw_by_match.setdefault(match_id, [])
        for market in event.get("markets") or []:
            ticker = str(market.get("ticker") or "").strip()
            if not ticker:
                continue
            series = str(market.get("series_ticker") or event.get("series_ticker") or "")
            market = {**market, "series_ticker": series}
            kind = _market_type(market, series)
            p_yes, reason = model_market(market, series, game, prediction)
            line = _line_from_market(market)
            yes_team = _market_yes_team(market, home, away)
            catalog.append({
                "event_ticker": event.get("event_ticker", ""), "ticker": ticker,
                "series_ticker": series, "event_title": " ".join(str(event.get(k) or "") for k in ("title", "sub_title")),
                "market_title": market.get("yes_sub_title") or market.get("title") or ticker,
                "market_status": market.get("status") or "", "yes_bid_dollars": market.get("yes_bid_dollars"),
                "yes_ask_dollars": market.get("yes_ask_dollars"), "no_bid_dollars": market.get("no_bid_dollars"),
                "no_ask_dollars": market.get("no_ask_dollars"), "modelable": "SI" if valid_probability(p_yes) is not None else "NO",
                "motivo": "" if valid_probability(p_yes) is not None else reason,
            })
            if valid_probability(p_yes) is None:
                rejected.append({"match_id": match_id, "local": home, "visita": away, "event_ticker": event.get("event_ticker", ""), "ticker": ticker, "categoria": kind, "motivo": reason})
                continue
            for side, p in (("YES", p_yes), ("NO", 1 - p_yes)):
                price = _price(market, side)
                edge = p - price if price is not None else None
                description, yes_outcome, description_team = _description(market, kind, home, away, side, line)
                if not description:
                    rejected.append({"match_id": match_id, "local": home, "visita": away, "event_ticker": event.get("event_ticker", ""), "ticker": ticker, "categoria": kind, "motivo": "no se pudo generar una frase de apuesta inequívoca"})
                    continue
                captured = now.isoformat()
                raw_by_match[match_id].append({
                    "schema_version": SCHEMA_VERSION, "prediction_id": "",
                    "event_ticker": event.get("event_ticker", ""), "ticker": ticker,
                    "match_id": match_id, "fecha": kickoff.isoformat(), "hora_inicio": kickoff.isoformat(),
                    "local": home, "visita": away, "categoria": kind,
                    "mercado": market.get("yes_sub_title") or market.get("title") or ticker,
                    "line": line if line is not None else "", "lado": side, "seleccion": side,
                    "equipo_yes": yes_team or description_team, "tipo_salida": yes_outcome, "descripcion": description,
                    "bloque": "", "probabilidad_original": f"{p:.8f}", "probabilidad_calibrada": "",
                    "version_calibracion": "calibración insuficiente", "probabilidad_modelo": f"{p:.8f}",
                    "precio_captura": f"{price:.6f}" if price is not None else "",
                    "probabilidad_implicita": f"{price:.6f}" if price is not None else "",
                    "edge": f"{edge:.6f}" if edge is not None else "",
                    "revision": "REVISAR" if edge is not None and edge > 0.20 else "",
                    "estado_modelo": "NFL Elo + forma reciente; calibración insuficiente",
                    "reglas": market.get("rules_primary") or market.get("rules_secondary") or "",
                    "capturado_en": captured,
                    "disponibilidad": "Disponible en Kalshi" if price is not None else "Sin precio en Kalshi",
                })

    saved_rows, coverage = [], []
    min_combo = float(os.getenv("NFL_PROBABILIDAD_MINIMA_COMBO", "0.65"))
    for match_id, info in match_info.items():
        raw_rows = raw_by_match.get(match_id, [])
        candidates = []
        for row in raw_rows:
            p = valid_probability(row.get("probabilidad_modelo"))
            if p is None:
                continue
            try:
                price = float(row.get("precio_captura")) if row.get("precio_captura") else None
            except (TypeError, ValueError):
                price = None
            try:
                edge = float(row.get("edge")) if row.get("edge") else None
            except (TypeError, ValueError):
                edge = None
            candidates.append({
                **row, "_p": p, "_lado": row["lado"], "Ticker": row["ticker"],
                "Apuesta": row["descripcion"], "Categoría resumen": row["categoria"],
                "Probabilidad del modelo": f"{p:.1%}",
                "Precio de compra": f"{price:.0%}" if price is not None else "—",
                "Ventaja (edge)": edge, "Lado Kalshi": row["lado"],
            })
        singles = rank_single_candidates(candidates, limit=5)
        high_all, value_all = rank_market_candidates(
            candidates, min_probability=min_combo,
            high_limit=max(1, len(candidates)), value_limit=max(1, len(candidates)),
        )
        high = sorted(high_all, key=_edge_number, reverse=True)[:5]
        # The shared ranking enforces line diversification; this final guard also enforces one pick per family.
        value, families = [], set()
        for candidate in sorted(value_all, key=_edge_number, reverse=True):
            family = _market_family(candidate)
            if family in families:
                continue
            families.add(family)
            value.append(candidate)
            if len(value) >= 10:
                break
        selected = {
            "SENCILLAS": singles,
            "ALTA_PROBABILIDAD": high,
            "A_TENER_EN_CUENTA": value,
        }
        key_to_raw = {(row["ticker"], row["lado"]): row for row in raw_rows}
        for block, chosen in selected.items():
            for candidate in chosen:
                key = (str(candidate.get("ticker") or candidate.get("Ticker") or ""), str(candidate.get("lado") or candidate.get("_lado") or "").upper())
                raw = key_to_raw.get(key)
                if not raw:
                    continue
                row = dict(raw)
                row["bloque"] = block
                row["prediction_id"] = hashlib.sha256(
                    f"{row['ticker']}|{row['lado']}|{block}|{row['capturado_en']}".encode()
                ).hexdigest()[:24]
                saved_rows.append(row)
        reasons = single_candidate_rejection_reasons(candidates)
        coverage.append({
            "match_id": match_id, **info, "contratos_modelables": len(candidates),
            "sencillas_validas": len(singles), "alta_probabilidad_validas": len(high),
            "a_tener_en_cuenta_validas": len(value),
            "motivo_sencillas": "" if singles else _reason_for_empty("SENCILLAS", candidates),
            "motivo_alta_probabilidad": "" if high else _reason_for_empty("ALTA_PROBABILIDAD", candidates),
            "motivo_a_tener_en_cuenta": "" if value else _reason_for_empty("A_TENER_EN_CUENTA", candidates),
        })
        selected_keys = {(str(c.get("ticker") or c.get("Ticker") or ""), str(c.get("lado") or c.get("_lado") or "").upper()) for chosen in selected.values() for c in chosen}
        for candidate in candidates:
            key = (str(candidate["ticker"]), str(candidate["lado"]).upper())
            if key not in selected_keys:
                rejected.append({
                    "match_id": match_id, "local": info["local"], "visita": info["visita"],
                    "event_ticker": candidate.get("event_ticker", ""), "ticker": candidate["ticker"],
                    "categoria": candidate["categoria"],
                    "motivo": "excluida por filtros/diversificación/límite; " + "; ".join(f"{k}={v}" for k, v in reasons.items()),
                })
    return saved_rows, rejected, catalog, coverage

def save_capture(rows, path=PRED_FILE):
    """Shared immutable writer wrapped in an atomic file replacement."""
    import shutil
    path = Path(path)
    tmp = path.with_name(path.name + ".capture.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copyfile(path, tmp)
    elif tmp.exists():
        tmp.unlink()
    result = append_immutable_predictions(
        tmp, PRED_COLUMNS, rows, probability_field="probabilidad_modelo",
        kickoff_field="hora_inicio", capture_field="capturado_en",
    )
    os.replace(tmp, path)
    return result

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
    rows, rejected, catalog, coverage = generate_candidates(events, model, upcoming_games=upcoming)
    # Archive full observed catalog and rejection reasons for auditability.
    _write_report(CATALOG_REPORT,
        ["event_ticker", "ticker", "series_ticker", "event_title", "market_title", "market_status",
         "yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars", "modelable", "motivo"], catalog)
    _write_report(REJECTED_REPORT,
        ["match_id", "local", "visita", "event_ticker", "ticker", "categoria", "motivo"], rejected)
    _write_report(COVERAGE_REPORT, COVERAGE_COLUMNS, coverage)
    saved = save_capture(rows)
    print(f"NFL catálogo: series={len(series)} eventos={len(events)} mercados={len(catalog)} modelables={sum(x['modelable']=='SI' for x in catalog)} descartados={len(rejected)}")
    print(f"NFL captura: aceptadas={saved['accepted']} rechazadas={saved['rejected']} histórico_modelo={model['trained_games']} duración_s={time.monotonic()-started:.1f}")
    return {"series": series, "events": events, "catalog": catalog, "saved": saved, "rejected": rejected}


def _espn_events_for_date(day):
    key = day.isoformat()
    if key not in ESPN_SCOREBOARD_CACHE:
        payload = _get(ESPN_SCOREBOARD_URL, {"dates": day.strftime("%Y%m%d"), "limit": 100})
        ESPN_SCOREBOARD_CACHE[key] = payload.get("events") or []
    return ESPN_SCOREBOARD_CACHE[key]


def settle_from_espn(row):
    """Independent fallback for supported full-game markets; refuses ambiguous fixtures/pushes."""
    kickoff = parse_dt(row.get("hora_inicio") or row.get("fecha"))
    home = _canonical_team_code(row.get("local"))
    away = _canonical_team_code(row.get("visita"))
    if not kickoff or home not in TEAM_ALIASES or away not in TEAM_ALIASES or home == away:
        return None
    base_date = kickoff.astimezone(TZ_ET).date()
    matches = []
    # Search a small window to handle official rescheduling; exact teams and unique fixture required.
    for delta in range(-3, 4):
        day = base_date + timedelta(days=delta)
        try:
            events_for_day = _espn_events_for_date(day)
        except (requests.RequestException, ValueError):
            continue
        for event in events_for_day:
            competitions = event.get("competitions") or []
            if not competitions:
                continue
            comp = competitions[0]
            teams = {}
            for competitor in comp.get("competitors") or []:
                side = str(competitor.get("homeAway") or "").lower()
                code = _canonical_team_code((competitor.get("team") or {}).get("abbreviation"))
                if side in {"home", "away"}:
                    teams[side] = code
            if teams.get("home") != home or teams.get("away") != away:
                continue
            game_dt = parse_dt(comp.get("date") or event.get("date"))
            if not game_dt or abs((game_dt - kickoff).total_seconds()) > 96 * 3600:
                continue
            status = (event.get("status") or {}).get("type") or {}
            status_name = str(status.get("name") or "").upper()
            if status_name in {"STATUS_CANCELED", "STATUS_CANCELLED"}:
                matches.append({"state": "ANULADA", "source": "ESPN", "reason": "partido oficialmente cancelado por ESPN", "score": ""})
                continue
            if not status.get("completed") and status_name not in {"STATUS_FINAL", "STATUS_FINAL_OT", "STATUS_FINAL_2OT"}:
                continue
            scores = {}
            for competitor in comp.get("competitors") or []:
                side = str(competitor.get("homeAway") or "").lower()
                try:
                    scores[side] = int(competitor.get("score"))
                except (TypeError, ValueError):
                    scores[side] = None
            if scores.get("home") is None or scores.get("away") is None:
                continue
            matches.append({
                "home_score": scores["home"], "away_score": scores["away"],
                "score": f"{away} {scores['away']} - {home} {scores['home']}",
                "state": "FINAL", "source": "ESPN", "reason": "marcador final ESPN (incluye prórroga cuando el marcador oficial la incluye)",
                "date": game_dt,
            })
    if len(matches) != 1:
        return None
    game = matches[0]
    if game["state"] == "ANULADA":
        return {"state": "ANULADA", "source": game["source"], "reason": game["reason"], "score": ""}
    home_score, away_score = game["home_score"], game["away_score"]
    category = str(row.get("categoria") or "")
    side = str(row.get("lado") or row.get("seleccion") or "").upper()
    if side not in {"YES", "NO"}:
        return None
    try:
        line = float(row.get("line"))
    except (TypeError, ValueError):
        line = None
    yes_team = _canonical_team_code(row.get("equipo_yes"))
    outcome = str(row.get("tipo_salida") or "")
    if category == "winner":
        if yes_team == home:
            yes_hit = home_score > away_score
        elif yes_team == away:
            yes_hit = away_score > home_score
        else:
            return None
        if home_score == away_score:
            return None
    elif category == "spread":
        if line is None or yes_team not in {home, away}:
            return None
        team_margin = (home_score - away_score) if yes_team == home else (away_score - home_score)
        if outcome == "win_by":
            yes_hit = team_margin >= math.ceil(abs(line))
        elif outcome == "spread_cover":
            yes_hit = team_margin + line > 0
            if team_margin + line == 0:
                return None
        else:
            return None
    elif category in {"total", "team_total"}:
        if line is None:
            return None
        if category == "total":
            score = home_score + away_score
        elif yes_team == home:
            score = home_score
        elif yes_team == away:
            score = away_score
        else:
            return None
        if outcome == "under_le":
            yes_hit = score <= math.floor(line)
        elif outcome == "under_lt":
            yes_hit = score < line
        elif outcome == "over_ge":
            yes_hit = score >= math.ceil(line)
        elif outcome == "over_gt":
            yes_hit = score > line
        else:
            return None
    else:
        return None
    won = yes_hit if side == "YES" else not yes_hit
    return {
        "state": "WIN" if won else "LOSS", "source": "ESPN",
        "reason": game["reason"], "score": game["score"],
    }


def settle():
    """Resolve all stored predictions from Kalshi first; conservative ESPN fallback for full-game markets."""
    import json
    import requests
    from prediction_core import kalshi_get
    predictions = []
    if PRED_FILE.exists():
        with PRED_FILE.open(newline="", encoding="utf-8-sig") as handle:
            predictions = list(csv.DictReader(handle))
    results = []
    if RESULT_FILE.exists():
        with RESULT_FILE.open(newline="", encoding="utf-8-sig") as handle:
            results = list(csv.DictReader(handle))
    else:
        _write_report(RESULT_FILE, RESULT_COLUMNS, [])
    latest = {}
    for row in results:
        pid = str(row.get("prediction_id") or "")
        if pid and (pid not in latest or str(row.get("resuelto_en", "")) > str(latest[pid].get("resuelto_en", ""))):
            latest[pid] = row
    new_results = []
    market_cache = {}
    for row in predictions:
        pid = str(row.get("prediction_id") or "")
        if not pid or str(latest.get(pid, {}).get("estado", "")).upper() in TERMINAL:
            continue
        ticker = str(row.get("ticker") or "")
        score = ""
        if ticker not in market_cache:
            try:
                market_cache[ticker] = kalshi_get(f"markets/{ticker}").get("market", {})
            except (requests.RequestException, ValueError, KeyError):
                market_cache[ticker] = {}
        market_data = market_cache[ticker]
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
            fallback = settle_from_espn(row)
            if not fallback or fallback.get("state") not in TERMINAL:
                continue
            state, source, reason = fallback["state"], fallback.get("source", "ESPN"), fallback.get("reason", "")
            score = fallback.get("score", "")
        new_results.append({
            "schema_version": SCHEMA_VERSION, "prediction_id": pid, "estado": state,
            "resultado_kalshi": kalshi_result, "marcador": score, "fuente": source,
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
        now = datetime.now(timezone.utc)
        game_events = [
            e for e in events
            if "GAME" in str(e.get("series_ticker") or e.get("event_ticker") or "").upper()
            and len(_event_teams(e)) == 2
            and (e.get("markets") or [])
        ]
        example = next(({
            "event_ticker": e.get("event_ticker"),
            "series_ticker": e.get("series_ticker") or str(e.get("event_ticker") or "").split("-")[0],
            "title": e.get("title") or e.get("sub_title"),
            "subtitle": e.get("sub_title") or e.get("subtitle"),
            "start_time": _kickoff(e).isoformat() if _kickoff(e) else None,
            "teams_detected": _event_teams(e),
            "markets": [
                {
                    "ticker": m.get("ticker"),
                    "series_ticker": m.get("series_ticker") or e.get("series_ticker"),
                    "title": m.get("title") or m.get("yes_sub_title"),
                    "yes_ask_dollars": m.get("yes_ask_dollars"),
                    "no_ask_dollars": m.get("no_ask_dollars"),
                }
                for m in (e.get("markets") or [])[:12]
            ],
        } for e in game_events), None)
        if not example:
            diagnostics = [
                {
                    "event_ticker": e.get("event_ticker"),
                    "series_ticker": e.get("series_ticker"),
                    "title": e.get("title"),
                    "sub_title": e.get("sub_title"),
                    "market_count": len(e.get("markets") or []),
                }
                for e in events[:20]
            ]
            print(json.dumps({"series_count": len(series), "series": [{"ticker": x.get("ticker"), "title": x.get("title")} for x in series], "events_count": len(events), "sample_events": diagnostics}, indent=2))
            raise RuntimeError("La API de Kalshi no devolvió evento de partido con mercados anidados.")
        print(json.dumps({
            "series": [{"ticker": x.get("ticker"), "title": x.get("title")} for x in series],
            "open_event_count": len(events),
            "game_series_event_count": len(game_events),
            "example_event": example,
        }, indent=2))
