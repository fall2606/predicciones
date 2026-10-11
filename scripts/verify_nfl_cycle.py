"""Live NFL integration smoke: actual open Kalshi capture + freeze, and settled historical ticker resolution."""
from __future__ import annotations

import csv
import csv
import json
import tempfile
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nfl_engine as nfl
import prediction_core


def _write(path, columns, rows):
    nfl._write_report(path, columns, rows)


def _market_result(ticker):
    try:
        return nfl.kalshi_get(f"markets/{ticker}").get("market", {})
    except Exception:
        return {}


def _espn_fixture(event):
    event_date = nfl._event_date(event)
    teams = nfl._event_teams(event)
    if not event_date or len(teams) != 2:
        return None
    for scoreboard_event in nfl._espn_events_for_date(event_date):
        competitions = scoreboard_event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        sides = {}
        for competitor in comp.get("competitors") or []:
            side = str(competitor.get("homeAway") or "").lower()
            code = nfl._canonical_team_code((competitor.get("team") or {}).get("abbreviation"))
            if side in {"home", "away"}:
                sides[side] = code
        if len(sides) != 2 or set(sides.values()) != set(teams):
            continue
        kickoff = nfl.parse_dt(comp.get("date") or scoreboard_event.get("date"))
        if not kickoff:
            continue
        status = (scoreboard_event.get("status") or {}).get("type") or {}
        return {
            "home": sides["home"], "away": sides["away"], "date": kickoff.isoformat(),
            "status": str(status.get("name") or "").upper(),
            "status_detail": str(status.get("shortDetail") or status.get("detail") or ""),
            "scoreboard_event": scoreboard_event,
        }
    return None


def _is_overtime(fixture, historical_games, game_date):
    status_text = f"{fixture['status']} {fixture['status_detail']}".upper()
    if "OT" in status_text or "OVERTIME" in status_text:
        return True
    if "overtime" in historical_games.columns:
        rows = historical_games[
            (historical_games["gameday"].dt.date == game_date)
            & (historical_games["home_team"] == fixture["home"])
            & (historical_games["away_team"] == fixture["away"])
        ]
        if not rows.empty:
            value = str(rows.iloc[-1].get("overtime", "")).lower()
            return value in {"true", "1", "yes", "y"}
    return False


def _settled_events(series, max_pages=3):
    events = []
    for item in series:
        ticker = str(item.get("ticker") or "").upper()
        try:
            rows = nfl.iter_kalshi_pages(
                "events",
                {"limit": 200, "series_ticker": ticker, "status": "settled", "with_nested_markets": "true"},
                "events", max_pages,
            )
            events.extend(rows)
        except Exception as exc:
            print(f"SETTLED_SERIES_WARNING {ticker}: {exc}")
    unique = {}
    for event in events:
        key = str(event.get("event_ticker") or "")
        if key:
            unique[key] = event
    return list(unique.values())


def _historical_overtime_fixture(series, historical_games, now):
    candidates = [
        item for item in series
        if "GAME" in str(item.get("ticker") or "").upper()
        and not any(token in str(item.get("ticker") or "").upper() for token in ("GAMETD", "GAMEPASS", "GAMERUSH"))
    ]
    events = _settled_events(candidates, max_pages=10)
    events.sort(key=lambda event: nfl._event_date(event) or datetime.min.date(), reverse=True)
    for event in events:
        game_date = nfl._event_date(event)
        if not game_date:
            continue
        fixture = _espn_fixture(event)
        if not fixture or not fixture["status"].startswith("STATUS_FINAL"):
            continue
        if not _is_overtime(fixture, historical_games, game_date):
            continue
        status_text = f"{fixture['status']} {fixture['status_detail']}".upper()
        fixture["overtime_verified_by"] = (
            "ESPN status" if ("OT" in status_text or "OVERTIME" in status_text)
            else "nflverse games.csv overtime flag"
        )
        series_ticker = str(event.get("series_ticker") or event.get("event_ticker") or "").split("-")[0]
        for market in event.get("markets") or []:
            ticker = str(market.get("ticker") or "")
            if not ticker or nfl._market_type(market, series_ticker) != "winner":
                continue
            details = _market_result(ticker)
            result = str(details.get("result") or market.get("result") or "").lower()
            if result in {"yes", "no"}:
                return event, market, details, fixture, series_ticker
    return None


def main():
    now = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(prefix="nfl-cycle-") as temp:
        temp = Path(temp)
        live_predictions = temp / "predicciones_nfl_live.csv"
        live_results = temp / "resultados_nfl_live.csv"
        historical_predictions = temp / "predicciones_nfl_historico.csv"
        historical_results = temp / "resultados_nfl_historico.csv"

        games = nfl._nflverse_games()
        model = nfl.build_team_model(games, as_of=now)
        upcoming = nfl.load_upcoming_games(days=21, now=now)
        open_series, open_events = nfl.load_kalshi_catalog(status="open", max_pages=1)
        print(f"LIVE_KALSHI_CATALOG_OK series={len(open_series)} events={len(open_events)}")
        live_rows, rejected, catalog, coverage = nfl.generate_candidates(
            open_events, model, now=now, upcoming_games=upcoming
        )
        future_rows = [row for row in live_rows if nfl.parse_dt(row.get("hora_inicio")) and nfl.parse_dt(row.get("hora_inicio")) > now]
        if not future_rows:
            reasons = [c for c in coverage if c.get("motivo_sencillas") or c.get("motivo_alta_probabilidad") or c.get("motivo_a_tener_en_cuenta")]
            print("LIVE_CAPTURE_NO_QUALIFYING_BETS")
            print(json.dumps(reasons[:10], ensure_ascii=False, indent=2))
            raise RuntimeError("No hay predicciones que cumplan los filtros para un partido NFL futuro; no se inventa una apuesta para forzar el ciclo.")
        first_match = min(future_rows, key=lambda row: nfl.parse_dt(row["hora_inicio"]))
        selected_match = first_match["match_id"]
        live_rows = [row for row in future_rows if row["match_id"] == selected_match]
        print(f"LIVE_MATCH_EXAMPLE event={first_match.get('event_ticker')} ticker={first_match.get('ticker')} market={first_match.get('mercado')} kickoff={first_match.get('hora_inicio')}")
        _write(live_predictions, nfl.PRED_COLUMNS, [])
        saved = nfl.save_capture(live_rows, path=live_predictions)
        if saved.get("accepted", 0) <= 0:
            raise RuntimeError(f"El writer no aceptó las filas del partido próximo: {saved}")
        before = live_predictions.read_text(encoding="utf-8")
        print(f"LIVE_CAPTURE_OK match_id={selected_match} rows_accepted={saved['accepted']}")
        print("LIVE_PREDICTIONS_BEFORE_FREEZE")
        print(before)

        sample = dict(live_rows[0])
        kickoff = nfl.parse_dt(sample["hora_inicio"])
        sample["prediction_id"] = "freeze-test-" + sample["prediction_id"]
        sample["ticker"] = sample["ticker"] + "-FREEZE-TEST"
        sample["lado"] = "NO" if sample["lado"] == "YES" else "YES"
        sample["seleccion"] = sample["lado"]
        sample["capturado_en"] = (kickoff + timedelta(minutes=1)).isoformat()
        class FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                fixed = kickoff + timedelta(minutes=1)
                return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)
        original_datetime = prediction_core.datetime
        try:
            prediction_core.datetime = FrozenDateTime
            freeze_result = nfl.save_capture([sample], path=live_predictions)
        finally:
            prediction_core.datetime = original_datetime
        after = live_predictions.read_text(encoding="utf-8")
        if freeze_result.get("accepted", 0) != 0 or after != before:
            raise AssertionError(f"El writer permitió modificar el ledger después del kickoff: {freeze_result}")
        print(f"FREEZE_AFTER_KICKOFF_OK accepted={freeze_result.get('accepted')} rejected={freeze_result.get('rejected')}")
        print("LIVE_PREDICTIONS_AFTER_FREEZE")
        print(after)

        series = nfl.discover_nfl_series()
        historical = _historical_overtime_fixture(series, games, now)
        if not historical:
            raise RuntimeError("No se encontró un contrato ganador liquidado de un partido NFL con prórroga en el catálogo consultado.")
        event, market, details, fixture, series_ticker = historical
        game_date = nfl._event_date(event)
        kickoff = nfl.parse_dt(fixture["date"])
        home, away = fixture["home"], fixture["away"]
        model_as_of = datetime.combine(game_date, datetime.min.time(), tzinfo=timezone.utc)
        historical_model = nfl.build_team_model(games, as_of=model_as_of)
        forecast = nfl.predict_game(historical_model, home, away)
        game = {"home": home, "away": away, "date": kickoff.isoformat()}
        p_yes, reason = nfl.model_market(market, series_ticker, game, forecast)
        if nfl.valid_probability(p_yes) is None:
            raise RuntimeError(f"No se pudo calcular el mercado ganador de la prueba histórica: {reason}")
        yes_team = nfl._market_yes_team(market, home, away)
        row = {column: "" for column in nfl.PRED_COLUMNS}
        row.update({
            "schema_version": nfl.SCHEMA_VERSION,
            "prediction_id": "historical-verification-" + str(market.get("ticker")),
            "event_ticker": event.get("event_ticker", ""), "ticker": market.get("ticker", ""),
            "match_id": nfl._match_id(kickoff, home, away), "fecha": kickoff.isoformat(),
            "hora_inicio": kickoff.isoformat(), "local": home, "visita": away,
            "categoria": "winner", "mercado": market.get("yes_sub_title") or market.get("title") or market.get("ticker"),
            "line": "", "lado": "YES", "seleccion": "YES", "equipo_yes": yes_team or "",
            "tipo_salida": "winner", "descripcion": f"SÍ: {_team_label(yes_team)} ganan el partido" if yes_team else "SÍ: ganador",
            "bloque": "VERIFICACION_HISTORICA", "probabilidad_original": f"{p_yes:.8f}",
            "probabilidad_modelo": f"{p_yes:.8f}", "version_calibracion": "fixture walk-forward; sin snapshot de precio prepartido",
            "capturado_en": (kickoff - timedelta(minutes=10)).isoformat(),
            "disponibilidad": "Fixture histórico de liquidación; no es predicción de producción",
        })
        _write(historical_predictions, nfl.PRED_COLUMNS, [row])
        nfl.PRED_FILE, nfl.RESULT_FILE = historical_predictions, historical_results
        print("HISTORICAL_PREDICTIONS_BEFORE_SETTLEMENT")
        print(historical_predictions.read_text(encoding="utf-8"))
        settle_summary = nfl.settle()
        result_text = historical_results.read_text(encoding="utf-8")
        print("HISTORICAL_RESULTS_AFTER_SETTLEMENT")
        print(result_text)
        result_rows = list(csv.DictReader(result_text.splitlines()))
        if not result_rows or result_rows[-1].get("estado") not in nfl.TERMINAL:
            raise AssertionError(f"La liquidación del ticker real no llegó a estado terminal: {settle_summary}")
        print(
            f"HISTORICAL_OT_SETTLEMENT_OK event={event.get('event_ticker')} ticker={market.get('ticker')} "
            f"kalshi_result={details.get('result') or market.get('result')} state={result_rows[-1].get('estado')} "
            f"source={result_rows[-1].get('fuente')} status={fixture['status']} overtime_verified_by={fixture['overtime_verified_by']}"
        )

        player_series = [
            item for item in series
            if any(token in str(item.get("ticker") or "").upper() for token in ("PASSYDS", "RUSHYDS", "RECYDS", "RECEPTIONS", "PLAYER"))
        ]
        prop_events = _settled_events(player_series[:8], max_pages=1)
        prop_fixture = None
        for prop_event in prop_events:
            for prop_market in prop_event.get("markets") or []:
                prop_ticker = str(prop_market.get("ticker") or "")
                prop_series = str(prop_market.get("series_ticker") or prop_event.get("series_ticker") or "")
                if prop_ticker and nfl._market_type(prop_market, prop_series) == "unsupported_player_prop":
                    _, prop_reason = nfl.model_market(prop_market, prop_series, {}, {})
                    prop_fixture = (prop_event, prop_market, prop_reason, _market_result(prop_ticker))
                    break
            if prop_fixture:
                break
        if not prop_fixture:
            raise RuntimeError("No se encontró un prop de jugador real para comprobar que el modelo lo excluye si no puede calcularlo.")
        prop_event, prop_market, prop_reason, prop_details = prop_fixture
        print(
            f"PLAYER_PROP_REAL_REJECTED event={prop_event.get('event_ticker')} ticker={prop_market.get('ticker')} "
            f"reason={prop_reason} kalshi_result={prop_details.get('result') or prop_market.get('result')}"
        )

        void_fixture = None
        void_checks = 0
        for prop_event in prop_events:
            for prop_market in prop_event.get("markets") or []:
                void_checks += 1
                if void_checks > 100:
                    break
                prop_ticker = str(prop_market.get("ticker") or "")
                if not prop_ticker:
                    continue
                details = _market_result(prop_ticker)
                result = str(details.get("result") or prop_market.get("result") or "").lower()
                if result in {"void", "canceled", "cancelled", "annulled"}:
                    void_fixture = (prop_event, prop_market, details)
                    break
            if void_fixture or void_checks > 100:
                break
        if void_fixture:
            prop_event, prop_market, details = void_fixture
            void_row = {column: "" for column in nfl.PRED_COLUMNS}
            void_row.update({
                "schema_version": nfl.SCHEMA_VERSION, "prediction_id": "void-verification-" + str(prop_market.get("ticker")),
                "event_ticker": prop_event.get("event_ticker", ""), "ticker": prop_market.get("ticker", ""),
                "match_id": "NFL:historical:VOID", "fecha": now.isoformat(), "hora_inicio": now.isoformat(),
                "local": "", "visita": "", "categoria": "unsupported_player_prop", "lado": "YES", "seleccion": "YES",
                "probabilidad_modelo": "", "capturado_en": now.isoformat(),
            })
            rows = list(csv.DictReader(historical_predictions.read_text(encoding="utf-8").splitlines()))
            rows.append(void_row)
            _write(historical_predictions, nfl.PRED_COLUMNS, rows)
            nfl.settle()
            result_text = historical_results.read_text(encoding="utf-8")
            print("HISTORICAL_RESULTS_AFTER_VOID_CASE")
            print(result_text)
            if not any(r.get("prediction_id") == void_row["prediction_id"] and r.get("estado") == "ANULADA" for r in csv.DictReader(result_text.splitlines())):
                raise AssertionError("Kalshi publicó un caso void pero el liquidador no lo marcó ANULADA.")
            print(f"VOID_SETTLEMENT_OK ticker={prop_market.get('ticker')} state=ANULADA")
        else:
            print(f"VOID_SETTLEMENT_NOT_FOUND: no se encontró un prop NFL anulado entre {void_checks} mercados consultados; no se inventó un caso.")

        print("LIMITATION: los contratos liquidados no proporcionan snapshot prepartido fiable; el fixture histórico no se presenta como predicción real de producción.")


def _team_label(team):
    return nfl.TEAM_ALIASES.get(team, (str(team),))[0].title()


if __name__ == "__main__":
    main()
