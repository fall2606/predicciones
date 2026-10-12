"""Utilities shared by football pages to keep match state and upcoming lists consistent."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Any

FINAL = {"FINISHED", "AWARDED", "FINAL", "COMPLETED"}
CANCELLED = {"CANCELLED", "CANCELED", "POSTPONED"}
SUSPENDED = {"SUSPENDED"}
LIVE = {"IN_PLAY", "PAUSED", "LIVE", "INPLAY"}
SCHEDULED = {"SCHEDULED", "TIMED"}
TERMINAL = FINAL | CANCELLED | SUSPENDED


def _status(match: Mapping[str, Any]) -> str:
    return str(match.get("status") or "").strip().upper()


def _rank(match: Mapping[str, Any]) -> int:
    status = _status(match)
    score = match.get("score") or {}
    full_time = score.get("fullTime") or {}
    has_score = full_time.get("home") is not None and full_time.get("away") is not None
    if status in FINAL:
        return 100 if has_score else 90
    if status in CANCELLED:
        return 85
    if status in SUSPENDED:
        return 80
    if status in LIVE:
        return 70
    if status in SCHEDULED:
        return 20
    return 10


def merge_match_sources(*sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate matches, preserving the most authoritative/advanced state."""
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        for raw in source or ():
            match = dict(raw)
            match_id = str(match.get("id") or "").strip()
            if not match_id:
                continue
            old = merged.get(match_id)
            if old is None:
                merged[match_id] = match
                continue
            # Keep useful fields from both records; status/score comes from the
            # record with the more advanced state so stale schedules cannot undo finals.
            preferred, secondary = (match, old) if _rank(match) > _rank(old) else (old, match)
            combined = dict(secondary)
            combined.update(preferred)
            if not combined.get("score") or not (combined.get("score") or {}).get("fullTime"):
                if secondary.get("score"):
                    combined["score"] = secondary["score"]
            merged[match_id] = combined
    return list(merged.values())


def apply_reconciliation_statuses(
    matches: Iterable[Mapping[str, Any]], csv_path: str | Path, sport: str
) -> list[dict[str, Any]]:
    """Promote stale calendar states when reconciliation already records a terminal event."""
    statuses: dict[str, set[str]] = {}
    path = Path(csv_path)
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("deporte", "")).strip().casefold() != sport.casefold():
                        continue
                    match_id = str(row.get("match_id", "")).strip()
                    state = str(row.get("estado_partido", "")).strip().upper()
                    if match_id and state:
                        statuses.setdefault(match_id, set()).add(state)
        except (OSError, csv.Error, UnicodeError):
            pass

    result = []
    for raw in matches:
        match = dict(raw)
        states = statuses.get(str(match.get("id") or "").strip(), set())
        if states & FINAL:
            match["status"] = "FINISHED"
        elif states & CANCELLED:
            match["status"] = "CANCELLED"
        elif states & SUSPENDED and _status(match) not in FINAL | CANCELLED:
            match["status"] = "SUSPENDED"
        result.append(match)
    return result


def is_upcoming_match(match: Mapping[str, Any], now: datetime | None = None) -> bool:
    """A scheduled match is upcoming only if its kickoff is still in the future."""
    if _status(match) not in SCHEDULED:
        return False
    kickoff = match.get("utcDate")
    if not kickoff:
        return False
    try:
        kickoff_dt = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if kickoff_dt.tzinfo is None:
        kickoff_dt = kickoff_dt.replace(tzinfo=timezone.utc)
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    return kickoff_dt.astimezone(timezone.utc) >= now_dt.astimezone(timezone.utc)
