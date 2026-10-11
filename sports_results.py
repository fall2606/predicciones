"""Fast independent football score checks using ESPN's public scoreboard endpoint.

The endpoint is public but undocumented; keep request volume low and never settle a
market from it unless its semantics can be derived safely from the captured prediction.
Special markets (corners, cards, player props, first goal, half-time) fall back to Kalshi.
"""
from __future__ import annotations

import ast
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import requests

LEAGUES = {"KXLALIGA": "esp.1", "KXUCL": "uefa.champions"}
_CACHE = {}
_SESSION = requests.Session()
_TIMEOUT = 8
ALIASES = {
    "barca": "barcelona", "barcelona fc": "barcelona",
    "atleti": "atletico madrid", "atletico": "atletico madrid",
    "atletico de madrid": "atletico madrid", "atletico madrid": "atletico madrid",
    "real sociedad": "real sociedad", "real madrid cf": "real madrid",
    "athletic club": "athletic bilbao", "athletic": "athletic bilbao",
    "club brugge": "club brugge", "brugge": "club brugge", "brugge kv": "club brugge",
    "atalanta bc": "atalanta", "bayer leverkusen": "bayer leverkusen",
    "inter": "inter milan", "internazionale": "inter milan",
    "psg": "paris saint germain", "paris sg": "paris saint germain",
    "bayern": "bayern munich", "bayern munchen": "bayern munich",
    "sporting": "sporting cp", "sporting lisbon": "sporting cp",
    "psv": "psv eindhoven", "ajax": "ajax amsterdam",
}


def _norm(value):
    value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    value = re.sub(r"\b(fc|cf|afc|club|de|the)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    value = re.sub(r"\s+", " ", value)
    return ALIASES.get(value, value)


def _similar(a, b):
    aa, bb = _norm(a), _norm(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    if aa in bb or bb in aa:
        return min(len(aa), len(bb)) / max(len(aa), len(bb)) * 0.15 + 0.82
    return SequenceMatcher(None, aa, bb).ratio()


def _parse_date(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _scoreboard(league, day):
    key = (league, day)
    if key in _CACHE:
        return _CACHE[key]
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
    response = _SESSION.get(url, params={"dates": day, "limit": 100}, timeout=_TIMEOUT)
    response.raise_for_status()
    events = (response.json() or {}).get("events") or []
    _CACHE[key] = events
    return events



_FD_CACHE = {}


def _football_data_match(row):
    """Use the documented football-data.org free API when CLAVE is available.

    One cached season request serves all predictions in that competition/season,
    avoiding one HTTP request per prediction and preserving the regulation-time score.
    """
    token = os.getenv("CLAVE")
    match_id = str(row.get("match_id") or "").strip()
    kickoff = _parse_date(row.get("fecha"))
    if not token or not match_id.isdigit() or kickoff is None:
        return None
    ticker = str(row.get("ticker") or "").upper()
    key_text = str(row.get("clave_interna") or "").upper()
    competition = "PD" if ticker.startswith("KXLALIGA") or "KXLALIGA" in key_text else (
        "CL" if ticker.startswith("KXUCL") or "KXUCL" in key_text else ""
    )
    if not competition:
        return None
    season = kickoff.year if kickoff.month >= 7 else kickoff.year - 1
    cache_key = (competition, season)
    if cache_key not in _FD_CACHE:
        try:
            response = _SESSION.get(
                f"https://api.football-data.org/v4/competitions/{competition}/matches",
                headers={"X-Auth-Token": token},
                params={"season": season, "limit": 500},
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            matches = (response.json() or {}).get("matches") or []
            _FD_CACHE[cache_key] = {str(item.get("id")): item for item in matches if item.get("id") is not None}
        except (requests.RequestException, ValueError, TypeError):
            _FD_CACHE[cache_key] = {}
    item = _FD_CACHE[cache_key].get(match_id)
    if not item:
        return None
    status = str(item.get("status") or "").upper()
    score = item.get("score") or {}
    full = score.get("fullTime") or {}
    home, away = full.get("home"), full.get("away")
    if home is None or away is None:
        return {
            "confidence": 1.0, "event_id": match_id, "status": status,
            "completed": False, "status_detail": "football-data.org: " + status,
            "local_score": None, "away_score": None, "league": competition,
            "event_name": "", "source": "football-data.org",
        }
    return {
        "confidence": 1.0, "event_id": match_id,
        "status": "STATUS_FINAL" if status == "FINISHED" else status,
        "completed": status == "FINISHED",
        "status_detail": "Marcador reglamentario de football-data.org: " + status,
        "local_score": int(home), "away_score": int(away),
        "league": competition, "event_name": "", "source": "football-data.org",
    }


def find_match(row):
    """Return a confidently matched scoreboard event, or None rather than guessing."""
    ticker = str(row.get("ticker") or "").upper()
    key = str(row.get("clave_interna") or "").upper()
    league = next((slug for prefix, slug in LEAGUES.items() if ticker.startswith(prefix) or prefix in key), None)
    if not league:
        return None
    kickoff = _parse_date(row.get("fecha"))
    if kickoff is None:
        return None
    local, away = str(row.get("local") or ""), str(row.get("visita") or "")
    if not local or not away:
        return None
    days = [(kickoff + timedelta(days=offset)).strftime("%Y%m%d") for offset in (0, -1, 1)]
    candidates = []
    for day in days:
        try:
            events = _scoreboard(league, day)
        except (requests.RequestException, ValueError, TypeError):
            continue
        for event in events:
            comps = event.get("competitions") or []
            competitors = (comps[0].get("competitors") or []) if comps else []
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away_team = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away_team:
                continue
            home_name = (home.get("team") or {}).get("displayName") or (home.get("team") or {}).get("name") or ""
            away_name = (away_team.get("team") or {}).get("displayName") or (away_team.get("team") or {}).get("name") or ""
            direct = (_similar(local, home_name) + _similar(away, away_name)) / 2
            reverse = (_similar(local, away_name) + _similar(away, home_name)) / 2
            score = max(direct, reverse)
            if score < 0.78:
                continue
            # A reversed home/away mapping is accepted only when both teams still match well.
            if direct >= reverse:
                local_score, away_score = home.get("score"), away_team.get("score")
            else:
                local_score, away_score = away_team.get("score"), home.get("score")
            status = (event.get("status") or {}).get("type") or {}
            candidates.append({
                "confidence": score, "event_id": str(event.get("id") or ""),
                "status": str(status.get("name") or status.get("state") or "").upper(),
                "completed": bool(status.get("completed")),
                "status_detail": str(status.get("detail") or status.get("description") or ""),
                "local_score": local_score, "away_score": away_score,
                "league": league, "event_name": str(event.get("name") or ""), "source": "ESPN",
            })
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    if len(candidates) > 1 and candidates[0]["confidence"] - candidates[1]["confidence"] < 0.025:
        return None
    best = candidates[0]
    if best["local_score"] in (None, "") or best["away_score"] in (None, ""):
        return best
    try:
        best["local_score"] = int(best["local_score"])
        best["away_score"] = int(best["away_score"])
    except (ValueError, TypeError):
        return None
    return best


def settle_from_score(row):
    """Resolve only full-time game result, full-time goals, BTTS, and team-goal totals."""
    # Prefer the documented, match-ID-based score source; use ESPN's public
    # scoreboard only when the official API/token is unavailable or has no match.
    match = _football_data_match(row)
    known_incident = match and any(
        token in str(match.get("status", "")).upper()
        for token in ("POSTPONED", "SUSPENDED", "CANCELLED", "CANCELED", "ABANDONED")
    )
    if not match or (not match.get("completed") and not known_incident):
        espn_match = find_match(row)
        if espn_match and espn_match.get("completed"):
            match = espn_match
        elif match is None:
            match = espn_match
    if not match:
        return None
    status = match["status"]
    if any(token in status for token in ("POSTPONED", "SUSPENDED", "CANCELLED", "CANCELED", "ABANDONED")):
        return {"match": match, "state": None, "reason": "estado_partido:" + status}
    if not match["completed"] or not any(token in status for token in ("FINAL", "FULL_TIME", "STATUS_FINAL")):
        return {"match": match, "state": None, "reason": "partido_no_finalizado:" + status}
    # Kalshi football contracts generally use regulation time; defer knockout games
    # that ESPN labels as extra-time or penalties to Kalshi's own settlement.
    if re.search(r"extra.?time|penalt|\baet\b|after extra", match.get("status_detail", ""), re.I):
        return None
    try:
        parsed = ast.literal_eval(str(row.get("clave_interna") or ""))
        if not (isinstance(parsed, tuple) and len(parsed) == 2 and isinstance(parsed[0], tuple)):
            return None
        contract, yes_no = parsed
        if len(contract) < 3 or str(yes_no).upper() not in {"YES", "NO"}:
            return None
        series, outcome = str(contract[0]).upper(), str(contract[1]).lower()
        line = contract[2]
        home, away = match["local_score"], match["away_score"]
        total = home + away
        proposition = None
        if series.endswith("GAME") and outcome in {"local", "visita", "empate"}:
            proposition = home > away if outcome == "local" else away > home if outcome == "visita" else home == away
        elif series.endswith("BTTS") and outcome in {"ambos", "btts"}:
            proposition = home > 0 and away > 0
        elif series.endswith("TEAMTOTAL") and outcome in {"local", "visita", "home", "away"} and line is not None:
            team_score = home if outcome in {"local", "home"} else away
            threshold = float(line)
            proposition = team_score > threshold
            if team_score == threshold:
                return None
        elif series.endswith("TOTAL") and outcome in {"over", "under"} and line is not None:
            threshold = float(line)
            proposition = total > threshold if outcome == "over" else total < threshold
            if total == threshold:
                # Whole-number lines can push; defer to Kalshi's exact market resolution.
                return None
        else:
            return None
        won = proposition if str(yes_no).upper() == "YES" else not proposition
        state = "WIN" if won else "LOSS"
        return {
            "match": match, "state": state,
            "reason": f"Marcador independiente ({match.get('source', 'ESPN')}): {home}-{away} (mercado {series}, selección {yes_no})",
        }
    except (ValueError, TypeError, SyntaxError, IndexError):
        return None
