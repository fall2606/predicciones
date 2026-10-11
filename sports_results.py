"""Independent football settlement using football-data.org and ESPN score/boxscore data.

Only resolve a market when the fixture, period, line and captured side can be matched
unambiguously. Unsupported stats and ambiguous contracts deliberately defer to Kalshi.
"""
from __future__ import annotations

import ast
import os
import re
import unicodedata
import time
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
    cached = _CACHE.get(key)
    if cached and time.monotonic() - cached[0] < 180:
        return cached[1]
    _CACHE.pop(key, None)
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
    response = _SESSION.get(url, params={"dates": day, "limit": 100}, timeout=_TIMEOUT)
    response.raise_for_status()
    events = (response.json() or {}).get("events") or []
    _CACHE[key] = (time.monotonic(), events)
    return events



_FD_CACHE = {}


def _football_data_match(row):
    """Match the fixture by kickoff and team names; app match_id is not the API's match id."""
    token = os.getenv("CLAVE")
    kickoff = _parse_date(row.get("fecha"))
    if not token or kickoff is None:
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
    cached = _FD_CACHE.get(cache_key)
    if cached and time.monotonic() - cached[0] < 180:
        matches = cached[1]
    else:
        _FD_CACHE.pop(cache_key, None)
        try:
            response = _SESSION.get(
                f"https://api.football-data.org/v4/competitions/{competition}/matches",
                headers={"X-Auth-Token": token},
                params={"season": season, "limit": 500},
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            matches = (response.json() or {}).get("matches") or []
        except (requests.RequestException, ValueError, TypeError):
            matches = []
        _FD_CACHE[cache_key] = (time.monotonic(), matches)
    local, away = str(row.get("local") or ""), str(row.get("visita") or "")
    if not local or not away:
        return None
    candidates = []
    for item in matches:
        home_team = (item.get("homeTeam") or {}).get("name") or (item.get("homeTeam") or {}).get("shortName") or ""
        away_team = (item.get("awayTeam") or {}).get("name") or (item.get("awayTeam") or {}).get("shortName") or ""
        match_date = _parse_date(item.get("utcDate"))
        if match_date is None or abs((match_date - kickoff).total_seconds()) > 36 * 3600:
            continue
        direct = (_similar(local, home_team) + _similar(away, away_team)) / 2
        reverse = (_similar(local, away_team) + _similar(away, home_team)) / 2
        confidence = max(direct, reverse)
        if confidence < 0.78:
            continue
        reversed_teams = reverse > direct
        score = item.get("score") or {}
        full = score.get("fullTime") or {}
        half = score.get("halfTime") or {}
        home_score, away_score = full.get("home"), full.get("away")
        half_home, half_away = half.get("home"), half.get("away")
        if reversed_teams:
            home_score, away_score = away_score, home_score
            half_home, half_away = half_away, half_home
        status = str(item.get("status") or "").upper()
        candidates.append({
            "confidence": confidence, "event_id": str(item.get("id") or ""),
            "status": "STATUS_FINAL" if status == "FINISHED" else status,
            "completed": status == "FINISHED",
            "status_detail": "Marcador reglamentario de football-data.org: " + status,
            "local_score": int(home_score) if home_score is not None else None,
            "away_score": int(away_score) if away_score is not None else None,
            "half_local_score": int(half_home) if half_home is not None else None,
            "half_away_score": int(half_away) if half_away is not None else None,
            "league": competition, "event_name": f"{home_team} - {away_team}", "source": "football-data.org",
        })
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    if len(candidates) > 1 and candidates[0]["confidence"] - candidates[1]["confidence"] < 0.025:
        return None
    return candidates[0]




_SUMMARY_CACHE = {}

def _espn_summary(event_id, league):
    """Fetch ESPN's detailed box score/plays for a confidently matched event."""
    if not event_id or not league:
        return None
    key = (str(league), str(event_id))
    cached = _SUMMARY_CACHE.get(key)
    if cached and time.monotonic() - cached[0] < 180:
        return cached[1]
    try:
        response = _SESSION.get(
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary",
            params={"event": str(event_id)}, timeout=_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json() or {}
    except (requests.RequestException, ValueError, TypeError):
        return None
    _SUMMARY_CACHE[key] = (time.monotonic(), payload)
    return payload


def _summary_first_scoring_team(summary):
    """Return the first scoring team's name only when ESPN exposes an ordered scorer list."""
    payload = summary or {}
    plays = payload.get("scoringPlays")
    if plays is None:
        plays = (payload.get("plays") or {}).get("items")
    if not isinstance(plays, list) or not plays:
        return None
    # The summary's scoringPlays list is already in match order.
    for play in plays:
        if not isinstance(play, dict):
            continue
        team = play.get("team") or (play.get("competitor") or {}).get("team") or {}
        name = team.get("displayName") or team.get("name") or team.get("shortDisplayName") or ""
        if name:
            return str(name)
    return None


def _summary_team_corners(summary, local_name, away_name):
    """Return (local, away) corners only when both teams and stats match uniquely."""
    box = (summary or {}).get("boxscore") or {}
    teams = box.get("teams") or []
    found = {}
    for item in teams:
        team = item.get("team") or {}
        name = team.get("displayName") or team.get("name") or team.get("shortDisplayName") or ""
        confidence_local = _similar(local_name, name)
        confidence_away = _similar(away_name, name)
        side = "local" if confidence_local >= confidence_away else "away"
        confidence = max(confidence_local, confidence_away)
        if confidence < 0.78:
            continue
        if side in found and found[side][0] >= confidence:
            return None
        value = None
        for stat in item.get("statistics") or []:
            label = str(stat.get("label") or stat.get("name") or stat.get("abbreviation") or "").lower()
            if "corner" not in label:
                continue
            raw = stat.get("displayValue", stat.get("value"))
            try:
                value = int(float(str(raw).replace(",", "")))
            except (TypeError, ValueError):
                continue
            break
        if value is not None:
            found[side] = (confidence, value)
    if set(found) != {"local", "away"}:
        return None
    return found["local"][1], found["away"][1]


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
                local_comp, away_comp = home, away_team
            else:
                local_comp, away_comp = away_team, home
            local_score, away_score = local_comp.get("score"), away_comp.get("score")
            def first_period_score(competitor):
                lines = competitor.get("linescores") or []
                if not isinstance(lines, list) or not lines:
                    return None
                value = lines[0].get("value") if isinstance(lines[0], dict) else lines[0]
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return None
            half_local_score = first_period_score(local_comp)
            half_away_score = first_period_score(away_comp)
            status = (event.get("status") or {}).get("type") or {}
            candidates.append({
                "confidence": score, "event_id": str(event.get("id") or ""),
                "status": str(status.get("name") or status.get("state") or "").upper(),
                "completed": bool(status.get("completed")),
                "status_detail": str(status.get("detail") or status.get("description") or ""),
                "local_score": local_score, "away_score": away_score,
                "half_local_score": half_local_score, "half_away_score": half_away_score,
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
        period_label = ""
        shown_home, shown_away = home, away
        half_home = match.get("half_local_score")
        half_away = match.get("half_away_score")
        proposition = None
        if series.endswith("GAME") and outcome in {"local", "visita", "empate"}:
            proposition = home > away if outcome == "local" else away > home if outcome == "visita" else home == away
        elif series.endswith("FTTS") and outcome in {"local", "visita", "home", "away"}:
            # A team-to-score-first contract is false when the final score is 0-0.
            # Otherwise require ESPN's ordered scoring-play list; never infer the scorer from score alone.
            if total == 0:
                proposition = False
            else:
                espn_match = find_match(row)
                if not espn_match or not espn_match.get("event_id"):
                    return None
                summary = _espn_summary(espn_match["event_id"], espn_match.get("league"))
                first_team = _summary_first_scoring_team(summary)
                if not first_team:
                    return None
                if _similar(first_team, str(row.get("local") or "")) >= 0.78:
                    first_side = "local"
                elif _similar(first_team, str(row.get("visita") or "")) >= 0.78:
                    first_side = "visita"
                else:
                    return None
                proposition = first_side == ("local" if outcome in {"local", "home"} else "visita")
        elif series.endswith("SPREAD") and line is not None:
            # Settle only explicit Spanish descriptions whose exact handicap semantics
            # are preserved in the captured row; otherwise defer to Kalshi.
            description = str(row.get("jugada") or row.get("mercado") or "").lower()
            spread_home, spread_away = home, away
            if "1.ª parte" in description or "1a parte" in description or "primera parte" in description:
                if half_home is None or half_away is None:
                    return None
                spread_home, spread_away = half_home, half_away
                period_label = "1.ª parte"
            match_desc = re.search(r"(.+?)\s+no pierde por más de\s+([0-9]+(?:[.,][0-9]+)?)\s+gol", description)
            if not match_desc:
                return None
            team_name = match_desc.group(1).strip()
            if _similar(team_name, str(row.get("local") or "")) >= 0.78:
                margin = spread_home - spread_away
            elif _similar(team_name, str(row.get("visita") or "")) >= 0.78:
                margin = spread_away - spread_home
            else:
                return None
            threshold = float(match_desc.group(2).replace(",", "."))
            proposition = margin >= -threshold
        elif series.endswith("1HBTTS") and outcome in {"ambos", "btts"}:
            if half_home is None or half_away is None:
                return None
            proposition = half_home > 0 and half_away > 0
            period_label = "1.ª parte"
            shown_home, shown_away = half_home, half_away
        elif series.endswith("BTTS") and outcome in {"ambos", "btts"}:
            proposition = home > 0 and away > 0
        elif series.endswith("TEAMTOTAL") and outcome in {"local", "visita", "home", "away"} and line is not None:
            team_score = home if outcome in {"local", "home"} else away
            threshold = float(line)
            proposition = team_score > threshold
            if team_score == threshold:
                return None
        elif series.endswith("1HTOTAL") and outcome in {"over", "under"} and line is not None:
            if half_home is None or half_away is None:
                return None
            shown_home, shown_away = half_home, half_away
            total = half_home + half_away
            threshold = float(line)
            proposition = total > threshold if outcome == "over" else total < threshold
            period_label = "1.ª parte"
            if total == threshold:
                return None
        elif series.endswith("2HTOTAL") and outcome in {"over", "under"} and line is not None:
            if half_home is None or half_away is None:
                return None
            shown_home, shown_away = home - half_home, away - half_away
            total = shown_home + shown_away
            threshold = float(line)
            proposition = total > threshold if outcome == "over" else total < threshold
            period_label = "2.ª parte"
            if total == threshold:
                return None
        elif series.endswith("TCORNERS") or series.endswith("CORNERS"):
            # Corners are not inferable from the score. Require ESPN's detailed
            # match boxscore and an unambiguous two-team match.
            espn_match = find_match(row)
            if not espn_match or not espn_match.get("completed") or not espn_match.get("event_id"):
                return None
            summary = _espn_summary(espn_match["event_id"], espn_match.get("league"))
            corners = _summary_team_corners(summary, str(row.get("local") or ""), str(row.get("visita") or ""))
            if corners is None or line is None:
                return None
            local_corners, away_corners = corners
            if series.endswith("TCORNERS") and outcome in {"local", "visita", "home", "away"}:
                count = local_corners if outcome in {"local", "home"} else away_corners
                proposition = count >= float(line)
                shown_home, shown_away = local_corners, away_corners
            elif series.endswith("CORNERS") and outcome in {"over", "under"}:
                count = local_corners + away_corners
                proposition = count >= float(line) if outcome == "over" else count < float(line)
                shown_home, shown_away = local_corners, away_corners
            else:
                return None
            period_label = "córners"
        elif series.endswith("TOTAL") and "1H" not in series and "2H" not in series and outcome in {"over", "under"} and line is not None:
            threshold = float(line)
            proposition = total > threshold if outcome == "over" else total < threshold
            if total == threshold:
                # Whole-number lines can push; defer to Kalshi's exact market resolution.
                return None
        else:
            return None
        won = proposition if str(yes_no).upper() == "YES" else not proposition
        state = "WIN" if won else "LOSS"
        label = f", {period_label}" if period_label else ""
        return {
            "match": match, "state": state,
            "reason": f"Marcador independiente ({match.get('source', 'ESPN')}{label}): {shown_home}-{shown_away} (mercado {series}, selección {yes_no})",
        }
    except (ValueError, TypeError, SyntaxError, IndexError):
        return None
