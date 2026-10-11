"""Live phase-1 smoke test: print a real NFL event and its Kalshi market catalog."""
import json
from nfl_engine import discover_nfl_catalog, example_catalog

catalog = discover_nfl_catalog()
example = example_catalog(catalog)
if not example:
    raise SystemExit("FAIL: Kalshi devolvió series NFL pero no eventos/mercados abiertos.")
print(json.dumps({
    "series_nfl": len(catalog["series"]),
    "eventos_nfl": len(catalog["events"]),
    "mercados_nfl": len(catalog["markets"]),
    "ejemplo_real": example,
}, ensure_ascii=False, indent=2))
