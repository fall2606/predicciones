from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import tempfile
from datetime import datetime, timedelta, timezone

from prediction_core import sports_series, events_for_series, iter_kalshi_pages, liquidate_kalshi_ledger


def settled_game_market(prefix, required_words=()):
    series = sports_series(prefixes=(prefix,))
    # Prefer the exact game series; do not test season-long futures.
    series.sort(key=lambda row: (row.get("ticker") != prefix, row.get("ticker", "")))
    for item in series[:8]:
        ticker = item.get("ticker")
        if not ticker:
            continue
        events = events_for_series([ticker], statuses=("settled",))
        # For the requested Barcelona-Getafe fixture, prefer that exact match if Kalshi retained it.
        if required_words:
            events = [e for e in events if all(w in str(e.get("title", "")).casefold() for w in required_words)]
        for event in events:
            for market in event.get("markets") or []:
                if market.get("ticker") and str(market.get("result", "")).lower() in {"yes", "no"}:
                    return ticker, event, market
        if required_words:
            continue
        try:
            markets = iter_kalshi_pages(
                "markets", {"limit": 200, "series_ticker": ticker, "status": "settled"},
                "markets", max_pages=3,
            )
        except Exception:
            markets = []
        for market in markets:
            if market.get("ticker") and str(market.get("result", "")).lower() in {"yes", "no"}:
                return ticker, {"title": market.get("event_ticker", "")}, market
    return None


def verify_real_settlement(label, prefix, required_words=()):
    found = settled_game_market(prefix, required_words)
    if not found:
        raise RuntimeError(f"Kalshi no devolvió un contrato de partido liquidado para {label} ({prefix}).")
    series_ticker, event, market = found
    result = str(market["result"]).upper()
    now = datetime.now(timezone.utc)
    past = (now - timedelta(days=1)).isoformat()
    with tempfile.TemporaryDirectory() as tmp:
        pred = Path(tmp) / "predicciones.csv"
        res = Path(tmp) / "resultados.csv"
        columns = ["prediction_id", "ticker", "match_id", "lado", "fecha", "capturado_en", "probabilidad_modelo"]
        with pred.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for side in ("YES", "NO"):
                writer.writerow({
                    "prediction_id": f"{label}_{side}", "ticker": market["ticker"],
                    "match_id": f"integration_{label}", "lado": side,
                    "fecha": past, "capturado_en": past, "probabilidad_modelo": "0.5",
                })
        result_columns = ["prediction_id", "estado", "marcador", "resuelto_en", "resultado_kalshi"]
        summary = liquidate_kalshi_ledger(pred, res, result_columns, side_field="lado")
        with res.open(newline="", encoding="utf-8") as f:
            rows = {row["prediction_id"]: row for row in csv.DictReader(f)}
        expected_win = "WIN" if result == "YES" else "LOSS"
        expected_loss = "LOSS" if result == "YES" else "WIN"
        if rows.get(f"{label}_YES", {}).get("estado") != expected_win:
            raise AssertionError(f"{label}: lado YES no coincide con resultado oficial {result}")
        if rows.get(f"{label}_NO", {}).get("estado") != expected_loss:
            raise AssertionError(f"{label}: lado NO no coincide con resultado oficial {result}")
        print(f"VERIFICADO {label}: serie={series_ticker}; evento={event.get('title')}; "
              f"ticker={market['ticker']}; resultado_Kalshi={result}; WIN/LOSS correctos; "
              f"liquidadas={summary['resultados_nuevos']}")


if __name__ == "__main__":
    # Recent LaLiga fixture requested by the user, plus a real settled MLB and UCL match.
    verify_real_settlement("LaLiga_Barcelona_Getafe", "KXLALIGAGAME", ("barcelona", "getafe"))
    verify_real_settlement("MLB_partido", "KXMLBGAME")
    verify_real_settlement("Champions_partido", "KXUCLGAME")
