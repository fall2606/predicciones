"""Live smoke test for the complete discoverable NFL Kalshi catalog."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_engine as nfl

series, events = nfl.load_kalshi_catalog(status="open", max_pages=2)
markets = [
    {"series_ticker": str(event.get("series_ticker") or ""),
     "event_ticker": str(event.get("event_ticker") or ""),
     "event_title": str(event.get("title") or ""),
     "ticker": str(market.get("ticker") or ""),
     "title": str(market.get("title") or market.get("yes_sub_title") or market.get("subtitle") or ""),
     "yes_ask_dollars": market.get("yes_ask_dollars"),
     "no_ask_dollars": market.get("no_ask_dollars"),
     "rules_primary": market.get("rules_primary") or "",
     "rules_secondary": market.get("rules_secondary") or ""}
    for event in events for market in (event.get("markets") or []) if market.get("ticker")
]
if not series:
    raise SystemExit("FAIL: Kalshi no devolvió series NFL.")
if not events or not markets:
    raise SystemExit("FAIL: no se encontró ningún evento/contrato abierto de NFL en Kalshi.")
sample_event = next(event for event in events if event.get("markets"))
sample_markets = [m for m in markets if m["event_ticker"] == sample_event.get("event_ticker")]
print(json.dumps({
    "series_nfl": len(series),
    "eventos_nfl": len(events),
    "mercados_nfl": len(markets),
    "ejemplo_real": {
        "event_ticker": sample_event.get("event_ticker"),
        "event_title": sample_event.get("title"),
        "markets": sample_markets[:15],
    },
}, ensure_ascii=False, indent=2))
