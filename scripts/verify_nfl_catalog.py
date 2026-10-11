"""Live smoke test showing every open Kalshi event/market found for one real NFL matchup."""
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_engine as nfl

ET = ZoneInfo("America/New_York")
series, events = nfl.load_kalshi_catalog(status="open", max_pages=2)
if not series:
    raise SystemExit("FAIL: Kalshi no devolvió series NFL.")
if not events:
    raise SystemExit("FAIL: Kalshi no devolvió eventos NFL abiertos.")

groups = {}
for event in events:
    teams = nfl._event_teams(event)
    day = nfl._event_date(event)
    if len(teams) != 2 or day is None:
        continue
    key = (day.isoformat(), tuple(sorted(teams)))
    groups.setdefault(key, []).append(event)

today_et = datetime.now(timezone.utc).astimezone(ET).date()
eligible = []
for key, grouped in groups.items():
    all_markets = [(event, market) for event in grouped for market in (event.get("markets") or []) if market.get("ticker")]
    families = {
        nfl._market_type(market, str(market.get("series_ticker") or event.get("series_ticker") or ""))
        for event, market in all_markets
    }
    has_winner = "winner" in families
    is_upcoming = datetime.fromisoformat(key[0]).date() >= today_et
    has_player_prop = "unsupported_player_prop" in families
    eligible.append((is_upcoming, has_winner, has_player_prop, len(all_markets), key, grouped, families))
if not eligible:
    raise SystemExit("FAIL: no se pudo agrupar un partido NFL por fecha ET y ambos equipos.")
# Prefer an upcoming matchup that has player props as well as a game-winner market.
eligible.sort(key=lambda item: (item[0] and item[1] and item[2], item[0] and item[1], item[2], item[3]), reverse=True)
upcoming_with_winner = [item for item in eligible if item[0] and item[1]]
sample = (upcoming_with_winner or eligible)[0]
_, _, _, _, key, grouped, families = sample
catalog = []
for event in grouped:
    for market in event.get("markets") or []:
        ticker = str(market.get("ticker") or "")
        if not ticker:
            continue
        st = str(market.get("series_ticker") or event.get("series_ticker") or "")
        catalog.append({
            "event_ticker": str(event.get("event_ticker") or ""),
            "event_title": str(event.get("title") or event.get("sub_title") or ""),
            "series_ticker": st,
            "ticker": ticker,
            "market_title": str(market.get("yes_sub_title") or market.get("title") or market.get("subtitle") or ""),
            "market_type": nfl._market_type(market, st),
            "line": nfl._line_from_market(market),
            "yes_ask_dollars": market.get("yes_ask_dollars"),
            "no_ask_dollars": market.get("no_ask_dollars"),
            "status": market.get("status"),
            "rules_primary": market.get("rules_primary") or "",
            "rules_secondary": market.get("rules_secondary") or "",
        })
catalog.sort(key=lambda item: (item["series_ticker"], item["ticker"]))
counts = Counter(item["market_type"] for item in catalog)
catalog_csv = Path(__file__).resolve().parents[1] / "nfl_catalogo_verificacion.csv"
csv_fields = ["fecha_partido_et", "equipos", "event_ticker", "event_title", "series_ticker", "ticker",
              "market_title", "market_type", "line", "yes_ask_dollars", "no_ask_dollars", "status",
              "rules_primary", "rules_secondary"]
with catalog_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    for item in catalog:
        writer.writerow({"fecha_partido_et": key[0], "equipos": " @ ".join(key[1]), **item})
print(json.dumps({
    "series_nfl_discovered": len(series),
    "open_events_nfl_scanned": len(events),
    "sample_match_date_et": key[0],
    "sample_teams": list(key[1]),
    "sample_event_count_across_series": len(grouped),
    "sample_market_count": len(catalog),
    "catalog_csv": str(catalog_csv),
    "market_families_found": dict(counts),
    "complete_match_catalog": catalog,
}, ensure_ascii=False, indent=2))
