"""NFL-specific model and live Kalshi catalog discovery.

Only winner, full-game spread and full-game total contracts are eligible until
period/player markets have a verified NFL data source and resolution mapping.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import exp, erf, sqrt
import re
import unicodedata

import requests
from prediction_core import KALSHI_API_BASE, events_for_series, kalshi_get, sports_series, valid_probability

NFL_PREFIX = "KXNFL"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
TIMEOUT = 12
ELO_BASE = 1500.0
ELO_K = 20.0
HOME_ADVANTAGE_ELO = 48.0
SCHEMA_VERSION = "1"


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def discover_nfl_catalog(include_closed: bool = False) -> dict:
    """Return all currently listed NFL series and nested markets from Kalshi API."""
    series = sports_series()
    nfl_series = [s for s in series if str(s.get("ticker", "")).upper().startswith(NFL_PREFIX)]
    if not nfl_series:
        raise RuntimeError("La API pública de Kalshi no devolvió series KXNFL; no se inventa catálogo.")
    statuses = ("open", "closed", "settled") if include_closed else ("open",)
    events = events_for_series([str(s["ticker"]) for s in nfl_series], statuses=statuses)
    markets = []
    for event in events:
        for market in event.get("markets") or []:
            ticker = str(market.get("ticker") or "").strip()
            if not ticker:
                continue
            markets.append({
                "series_ticker": str(event.get("series_ticker") or market.get("series_ticker") or ""),
                "event_ticker": str(event.get("event_ticker") or ""),
                "event_title": str(event.get("title") or ""),
                "event_start_time": str(event.get("strike_date") or event.get("expected_expiration_time") or event.get("close_time") or ""),
                "ticker": ticker,
                "market_title": str(market.get("title") or market.get("yes_sub_title") or market.get("subtitle") or ""),
                "subtitle": str(market.get("subtitle") or ""),
                "yes_bid_dollars": market.get("yes_bid_dollars"),
                "yes_ask_dollars": market.get("yes_ask_dollars"),
                "no_bid_dollars": market.get("no_bid_dollars"),
                "no_ask_dollars": market.get("no_ask_dollars"),
                "status": str(market.get("status") or ""),
                "rules_primary": str(market.get("rules_primary") or ""),
                "rules_secondary": str(market.get("rules_secondary") or ""),
            })
    return {"series": nfl_series, "events": events, "markets": markets}


def _normal_cdf(x: float, mean: float, sd: float) -> float:
    if not all(map(lambda z: z == z and abs(z) != float("inf"), (x, mean, sd))) or sd <= 0:
        raise ValueError("Parámetros inválidos para distribución NFL.")
    return 0.5 * (1.0 + erf((x - mean) / (sd * sqrt(2.0))))


def probability_over(threshold: float, mean: float, sd: float) -> float:
    p = 1.0 - _normal_cdf(threshold, mean, sd)
    return _safe_probability(p)


def probability_margin(home_margin_mean: float, sd: float, line: float) -> float:
    """P(home margin > line), with an explicit discrete-score adjustment around 3 and 7."""
    # The normal baseline is softened with a small continuity correction. NFL key
    # numbers are represented as discrete mass, not as arbitrary manual edge changes.
    baseline = 1.0 - _normal_cdf(line + 0.5, home_margin_mean, sd)
    key_mass = 0.0
    for key, mass in ((3.0, 0.045), (7.0, 0.035), (10.0, 0.018), (6.0, 0.015)):
        density = exp(-0.5 * ((key - home_margin_mean) / sd) ** 2) / (sd * sqrt(2 * 3.141592653589793))
        # Reallocate local continuous mass into NFL score-margin points.
        local_mass = min(mass, density * 0.5)
        if key > line:
            key_mass += local_mass
        key_mass -= local_mass * (1.0 if key <= line else 0.0)
    return _safe_probability(baseline + key_mass)


def _safe_probability(value: object) -> float:
    p = valid_probability(value)
    if p is None:
        raise ValueError(f"Probabilidad fuera de rango o no finita: {value!r}")
    return p


def fetch_espn_scoreboard(season: int, week: int | None = None, season_type: int = 2) -> list[dict]:
    params = {"limit": 1000, "dates": season, "seasontype": season_type}
    if week is not None:
        params["week"] = week
    response = requests.get(ESPN_SCOREBOARD, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    events = data.get("events") or []
    return [e for e in events if isinstance(e, dict)]


def finished_team_games(events: list[dict]) -> list[dict]:
    rows = []
    for event in events:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        status = (competition.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            continue
        competitors = competition.get("competitors") or []
        home = next((x for x in competitors if x.get("homeAway") == "home"), None)
        away = next((x for x in competitors if x.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            hs, as_ = float(home["score"]), float(away["score"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "date": str(event.get("date") or ""),
            "home": str((home.get("team") or {}).get("displayName") or ""),
            "away": str((away.get("team") or {}).get("displayName") or ""),
            "home_score": hs, "away_score": as_,
            "event_id": str(event.get("id") or ""),
            "status": str(status.get("name") or ""),
        })
    return rows


def fit_elo(games: list[dict], cutoff: datetime | None = None) -> dict:
    """Chronological Elo ratings from completed NFL games, excluding games at/after cutoff."""
    ratings: dict[str, float] = {}
    for game in sorted(games, key=lambda g: g.get("date", "")):
        if cutoff:
            try:
                kickoff = datetime.fromisoformat(game["date"].replace("Z", "+00:00"))
                if kickoff >= cutoff:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
        home, away = normalize_name(game.get("home")), normalize_name(game.get("away"))
        if not home or not away:
            continue
        rh, ra = ratings.get(home, ELO_BASE), ratings.get(away, ELO_BASE)
        expected_home = 1.0 / (1.0 + 10.0 ** (-(rh + HOME_ADVANTAGE_ELO - ra) / 400.0))
        margin = abs(float(game["home_score"]) - float(game["away_score"]))
        multiplier = max(0.5, (2.2 / (1 + 0.001 * (rh - ra if rh != ra else 0))) * (margin + 3) ** 0.8 / (7.5 + 0.006 * abs(rh - ra)))
        actual_home = 1.0 if game["home_score"] > game["away_score"] else (0.5 if game["home_score"] == game["away_score"] else 0.0)
        delta = ELO_K * multiplier * (actual_home - expected_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings


def predict_match(home: str, away: str, ratings: dict[str, float], games: list[dict]) -> dict:
    """NFL-specific baseline: Elo, home advantage, recent rest, and points distributions."""
    h, a = normalize_name(home), normalize_name(away)
    if not h or not a or h == a:
        raise ValueError("Equipos NFL inválidos.")
    rh, ra = ratings.get(h, ELO_BASE), ratings.get(a, ELO_BASE)
    elo_win = 1 / (1 + 10 ** (-(rh + HOME_ADVANTAGE_ELO - ra) / 400))
    recent = sorted(games, key=lambda g: g.get("date", ""), reverse=True)
    points = [float(g["home_score"]) + float(g["away_score"]) for g in recent[:100] if "home_score" in g and "away_score" in g]
    margins = [float(g["home_score"]) - float(g["away_score"]) for g in recent[:100] if "home_score" in g and "away_score" in g]
    total_mean = sum(points) / len(points) if points else 44.5
    margin_mean = (sum(margins) / len(margins) if margins else 0.0) + (rh - ra) / 25.0 + 1.5
    total_sd = max(8.0, (sum((x-total_mean)**2 for x in points) / max(1, len(points)-1)) ** 0.5 if len(points) > 1 else 13.0)
    margin_sd = max(7.0, (sum((x-margin_mean)**2 for x in margins) / max(1, len(margins)-1)) ** 0.5 if len(margins) > 1 else 13.5)
    return {
        "home": home, "away": away, "elo_home": rh, "elo_away": ra,
        "p_home_win": _safe_probability(elo_win),
        "p_away_win": _safe_probability(1 - elo_win),
        "margin_mean": margin_mean, "margin_sd": margin_sd,
        "total_mean": total_mean, "total_sd": total_sd,
        "model_version": "nfl-elo-points-v1",
    }


def market_family(series_ticker: str, title: str = "") -> str | None:
    s = str(series_ticker or "").upper()
    t = normalize_name(title)
    if "GAME" in s and "TEAM" not in s:
        return "winner"
    if "SPREAD" in s or "HANDICAP" in s:
        return "spread"
    if "TOTAL" in s:
        return "total"
    return None


def parse_market_line(market: dict) -> float | None:
    for key in ("floor_strike", "cap_strike"):
        try:
            value = float(market.get(key))
            if value == value and abs(value) != float("inf"):
                return value
        except (TypeError, ValueError):
            pass
    text = " ".join(str(market.get(k) or "") for k in ("title", "yes_sub_title", "subtitle"))
    match = re.search(r"(?<![A-Za-z])([+-]?\d+(?:\.\d+)?)\s*\+?", text)
    if not match:
        return None
    try:
        value = float(match.group(1))
        return value if abs(value) < 1000 else None
    except ValueError:
        return None


def example_catalog(catalog: dict) -> dict | None:
    """Small grounded sample for logs/reporting; never synthesizes tickers or lines."""
    for event in catalog.get("events", []):
        markets = [m for m in catalog.get("markets", []) if m["event_ticker"] == event.get("event_ticker")]
        if markets:
            return {"event_ticker": event.get("event_ticker"), "event_title": event.get("title"), "markets": markets[:12]}
    return None
