"""Single source of truth for terminal-prediction reconciliation across all sports."""
import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SPORTS = {
    "LaLiga": ("predicciones_laliga.csv", "resultados_laliga.csv", "football", "PD"),
    "Champions": ("predicciones_champions.csv", "resultados_champions.csv", "football", "CL"),
    "MLB": ("predicciones_mlb.csv", "resultados_mlb.csv", "mlb", ""),
    "NFL": ("predicciones_nfl.csv", "resultados_nfl.csv", "nfl", ""),
}
TERMINAL = {"WIN", "LOSS", "ANULADA"}
PLACEHOLDER_TICKERS = {"", "—", "-", "NONE", "NAN"}
PENDING_REPORT = ROOT / "reconciliacion_pendientes.csv"
PENDING_COLUMNS = [
    "deporte", "prediction_id", "match_id", "fecha", "local", "visita", "bloque",
    "ticker", "estado_registrado", "estado_partido", "estado_kalshi", "resultado_kalshi",
    "motivo", "clasificacion", "cuenta_alerta", "horas_desde_inicio", "resuelto_en",
]
GRACE_HOURS = max(1, int(os.getenv("KALSHI_SETTLEMENT_GRACE_HOURS", "72")))
MATCH_LOOKUP_GRACE_HOURS = max(1, int(os.getenv("MATCH_LOOKUP_GRACE_HOURS", "6")))
SESSION = requests.Session()
KALSHI_CACHE = {}


def rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def text(value):
    return str(value or "").strip()


def parse_dt(value):
    raw = text(value)
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def latest_results(result_rows):
    # A valid terminal outcome wins over any stale legacy NO_EVALUABLE/UNKNOWN row.
    terminal_rows = {}
    nonterminal_rows = {}
    for row in result_rows:
        pid = text(row.get("prediction_id"))
        if not pid:
            continue
        target = terminal_rows if text(row.get("estado")).upper() in TERMINAL else nonterminal_rows
        prev = target.get(pid)
        stamp = text(row.get("resuelto_en"))
        if prev is None or stamp >= text(prev.get("resuelto_en")):
            target[pid] = row
    return {**nonterminal_rows, **terminal_rows}


def finished_matches(kind, competition, predictions):
    now = datetime.now(timezone.utc)
    states = {}
    if kind == "mlb":
        dates = [parse_dt(r.get("fecha")) for r in predictions]
        dates = [d for d in dates if d]
        start = (min(dates).date() if dates else now.date()) - timedelta(days=3)
        end = now.date() + timedelta(days=1)
        response = SESSION.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "startDate": start.isoformat(), "endDate": end.isoformat(),
                    "gameTypes": "R,F,D,L,W"}, timeout=15,
        )
        response.raise_for_status()
        games = [g for day in response.json().get("dates", []) for g in day.get("games", [])]
        for game in games:
            if game.get("gamePk") is not None:
                status = game.get("status", {}) or {}
                states[str(game["gamePk"])] = text(status.get("detailedState") or status.get("abstractGameState")).upper()
        return states

    if kind == "nfl":
        # ESPN's public NFL scoreboard is the independent schedule/status source.
        dates = [parse_dt(r.get("fecha")) for r in predictions]
        dates = [d.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date() for d in dates if d]
        if not dates:
            dates = [now.date()]
        start, end = min(dates) - timedelta(days=3), max(dates) + timedelta(days=3)
        response = SESSION.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            params={"dates": f"{start:%Y%m%d}-{end:%Y%m%d}", "limit": 1000}, timeout=25,
        )
        response.raise_for_status()
        events = response.json().get("events", []) or []
        code_aliases = {"LA": "LAR", "JAC": "JAX", "OAK": "LV", "WFT": "WAS", "WSH": "WAS"}
        fixtures = []
        for event in events:
            competitions = event.get("competitions") or []
            if not competitions:
                continue
            comp = competitions[0]
            teams = {}
            for competitor in comp.get("competitors") or []:
                side = text(competitor.get("homeAway")).lower()
                code = text((competitor.get("team") or {}).get("abbreviation")).upper()
                code = code_aliases.get(code, code)
                if side in {"home", "away"}:
                    teams[side] = code
            kickoff = parse_dt(comp.get("date") or event.get("date"))
            if kickoff and teams.get("home") and teams.get("away"):
                status = (event.get("status") or {}).get("type") or {}
                fixtures.append({
                    "home": teams["home"], "away": teams["away"],
                    "date": kickoff.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date(),
                    "status": text(status.get("name") or status.get("description")).upper(),
                })
        for row in predictions:
            kickoff = parse_dt(row.get("fecha"))
            home = code_aliases.get(text(row.get("local")).upper(), text(row.get("local")).upper())
            away = code_aliases.get(text(row.get("visita")).upper(), text(row.get("visita")).upper())
            if not kickoff or not home or not away:
                continue
            target_date = kickoff.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date()
            matches = [
                g for g in fixtures if g["home"] == home and g["away"] == away
                and abs((g["date"] - target_date).days) <= 3
            ]
            if len(matches) == 1:
                states[text(row.get("match_id"))] = matches[0]["status"]
        return states

    token = os.getenv("CLAVE")
    if not token:
        raise RuntimeError("falta secreto CLAVE; no se puede comprobar el calendario oficial")
    # Query one season at a time; football-data rejects multi-year date ranges on some plans.
    seasons = set()
    for row in predictions:
        kickoff = parse_dt(row.get("fecha"))
        if kickoff:
            local_year = kickoff.year if kickoff.month >= 7 else kickoff.year - 1
            seasons.add(local_year)
    if not seasons:
        now = datetime.now(timezone.utc)
        seasons.add(now.year if now.month >= 7 else now.year - 1)
    for season in sorted(seasons):
        response = SESSION.get(
            "https://api.football-data.org/v4/competitions/" + competition + "/matches",
            headers={"X-Auth-Token": token}, params={"season": season, "limit": 500}, timeout=15,
        )
        response.raise_for_status()
        for match in response.json().get("matches", []):
            if match.get("id") is not None:
                states[str(match["id"])] = text(match.get("status")).upper()
    return states


def kalshi_market(ticker):
    if ticker in KALSHI_CACHE:
        return KALSHI_CACHE[ticker]
    try:
        response = SESSION.get(
            f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}", timeout=8
        )
        if response.status_code == 404:
            result = {"kind": "missing", "status": "not_found", "result": ""}
        else:
            response.raise_for_status()
            market = (response.json() or {}).get("market") or {}
            result = {
                "kind": "ok",
                "status": text(market.get("status")).lower(),
                "result": text(market.get("result")).lower(),
            }
    except (requests.RequestException, ValueError, TypeError) as exc:
        result = {"kind": "error", "status": type(exc).__name__, "result": ""}
    KALSHI_CACHE[ticker] = result
    return result


def captured_side(row):
    for key in ("lado", "seleccion"):
        side = text(row.get(key)).upper()
        if side in {"YES", "NO"}:
            return side
    raw = text(row.get("clave_interna"))
    if raw:
        try:
            import ast
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, tuple) and len(parsed) == 2:
                side = text(parsed[1]).upper()
                if side in {"YES", "NO"}:
                    return side
        except (ValueError, SyntaxError, TypeError):
            pass
    return None


def _row_report(name, row, latest, match_states, kind):
    pid = text(row.get("prediction_id"))
    match_id = text(row.get("match_id"))
    ticker_raw = text(row.get("ticker"))
    ticker = "" if ticker_raw.upper() in PLACEHOLDER_TICKERS else ticker_raw
    registered = latest.get(pid, {})
    state = text(registered.get("estado")).upper()
    kickoff = parse_dt(row.get("fecha"))
    now = datetime.now(timezone.utc)
    elapsed = round((now - kickoff).total_seconds() / 3600, 2) if kickoff else ""
    official_state = match_states.get(match_id, "")
    cancelled_states = {"POSTPONED", "CANCELLED", "CANCELED", "SUSPENDED", "STATUS_CANCELED", "STATUS_CANCELLED", "STATUS_POSTPONED", "STATUS_SUSPENDED"}
    final_states = {"FINISHED", "AWARDED", "FINAL", "COMPLETED", "STATUS_FINAL", "STATUS_FINAL_OT", "STATUS_FINAL_2OT"}
    is_final = official_state in final_states
    is_cancelled = official_state in cancelled_states
    grace_expired = bool(kickoff and now - kickoff >= timedelta(hours=GRACE_HOURS))
    match_lookup_expired = bool(kickoff and now - kickoff >= timedelta(hours=MATCH_LOOKUP_GRACE_HOURS))

    base = {
        "deporte": name, "prediction_id": pid, "match_id": match_id, "fecha": text(row.get("fecha")),
        "local": text(row.get("local")), "visita": text(row.get("visita")),
        "bloque": text(row.get("bloque")), "ticker": ticker_raw, "estado_registrado": state,
        "estado_partido": official_state or ("NO_ENCONTRADO" if match_id else "SIN_ID"),
        "estado_kalshi": "", "resultado_kalshi": "", "motivo": "", "clasificacion": "",
        "cuenta_alerta": "NO", "horas_desde_inicio": elapsed,
        "resuelto_en": text(registered.get("resuelto_en")),
    }

    if state in TERMINAL:
        base.update(motivo=f"resultado registrado: {state}", clasificacion="LIQUIDADA")
        return base

    # Explicitly preserve legacy predictions without a Kalshi contract, but never
    # count them as active unresolved contracts.
    if not ticker:
        base.update(motivo="sin ticker (histórico)", clasificacion="HISTORICO_SIN_CONTRATO")
        return base

    # Existing season/futures markets in old MLB captures are not game predictions.
    # Preserve them in the audit, but don't mislabel them as unfinished games.
    if not match_id and kickoff is None:
        base.update(motivo="partido no encontrado", clasificacion="HISTORICO_NO_PARTIDO")
        return base

    # Missing fixture IDs must not be silently skipped for 72 hours: after the
    # sport's normal duration, report the missing fixture; Kalshi gets its own 72h grace.
    if (not match_id or not official_state) and match_lookup_expired:
        market = kalshi_market(ticker)
        base["estado_kalshi"] = market.get("status", "")
        base["resultado_kalshi"] = market.get("result", "")
        if market["kind"] == "ok" and market["result"] in {"yes", "no"}:
            if captured_side(row) not in {"YES", "NO"}:
                base.update(motivo="estado desconocido: captura sin selección YES/NO; no se puede liquidar con seguridad",
                            clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
            else:
                base.update(motivo="resultado de Kalshi disponible pero no registrado",
                            clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
        elif market["kind"] == "ok" and market["result"] in {"void", "canceled", "cancelled"}:
            base.update(motivo="resultado de Kalshi disponible pero no registrado",
                        clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
        elif market["kind"] == "missing" or market.get("status") in {"closed", "inactive", "unlisted", "not_found"}:
            base.update(motivo="mercado sin resolución en Kalshi", clasificacion="HISTORICO_SIN_CONTRATO")
        elif market["kind"] == "ok" and market.get("status") in {"open", "active", "unopened", "initialized", "pending"}:
            if grace_expired:
                base.update(motivo=f"Kalshi aún no liquida (plazo vencido: {GRACE_HOURS} h)",
                            clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
            else:
                base.update(motivo="partido no encontrado", clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
        else:
            base.update(motivo="estado desconocido", clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
        return base

    if not (is_final or is_cancelled):
        # If an official fixture remains scheduled/in-play far past its start,
        # the status is stale or unknown and must be visible.
        if match_lookup_expired:
            base.update(motivo="estado desconocido", clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
        else:
            base.update(motivo="partido no finalizado", clasificacion="NO_FINALIZADA")
        return base

    if is_cancelled:
        # Liquidators should append ANULADA. If they did not, keep a visible actionable item.
        base.update(motivo="resultado de Kalshi disponible pero no registrado",
                    clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
        return base

    market = kalshi_market(ticker)
    base["estado_kalshi"] = market.get("status", "")
    base["resultado_kalshi"] = market.get("result", "")
    if market["kind"] == "missing":
        base.update(motivo="mercado sin resolución en Kalshi", clasificacion="HISTORICO_SIN_CONTRATO")
    elif market["kind"] == "error":
        base.update(motivo="estado desconocido", clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
    elif market["result"] in {"yes", "no"} and captured_side(row) not in {"YES", "NO"}:
        base.update(motivo="estado desconocido: captura sin selección YES/NO; no se puede liquidar con seguridad",
                    clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
    elif market["result"] in {"yes", "no", "void", "canceled", "cancelled"}:
        base.update(motivo="resultado de Kalshi disponible pero no registrado",
                    clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
    elif market["status"] in {"closed", "inactive", "unlisted", "not_found"}:
        base.update(motivo="mercado sin resolución en Kalshi", clasificacion="HISTORICO_SIN_CONTRATO")
    elif market["status"] in {"open", "active", "unopened", "initialized", "pending"}:
        if grace_expired:
            base.update(motivo=f"Kalshi aún no liquida (plazo vencido: {GRACE_HOURS} h)",
                        clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
        else:
            base.update(motivo=f"Kalshi aún no liquida (dentro del plazo: {GRACE_HOURS} h)",
                        clasificacion="EN_PLAZO")
    else:
        base.update(motivo="estado desconocido", clasificacion="PENDIENTE_ACTIVA", cuenta_alerta="SI")
    return base


def reconcile(name, pred_name, result_name, kind, competition):
    predictions = rows(ROOT / pred_name)
    results = rows(ROOT / result_name)
    latest = latest_results(results)
    match_states = finished_matches(kind, competition, predictions)
    report = []
    for row in predictions:
        report.append(_row_report(name, row, latest, match_states, kind))
    pending = [r for r in report if r["cuenta_alerta"] == "SI"]
    historical = [r for r in report if r["clasificacion"] == "HISTORICO_SIN_CONTRATO"]
    by_reason = {}
    for row in pending:
        by_reason[row["motivo"]] = by_reason.get(row["motivo"], 0) + 1
    print(f"RECONCILIACIÓN {name}: total={len(report)} | alerta={len(pending)} | "
          f"históricos_sin_contrato={len(historical)} | motivos={by_reason}")
    for row in pending:
        print("PENDIENTE", name, row["prediction_id"], row["match_id"], row["ticker"], row["motivo"])
    return report, len(pending)


def main():
    all_rows = []
    total_pending = 0
    errors = []
    for name, (pred, result, kind, competition) in SPORTS.items():
        try:
            report, pending = reconcile(name, pred, result, kind, competition)
            all_rows.extend(report)
            total_pending += pending
        except (requests.RequestException, ValueError, KeyError, RuntimeError, TypeError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"ERROR RECONCILIACIÓN {name}: {errors[-1]}")
            # Do not silently claim zero: surface a diagnostic row for each prediction
            # that could not be checked, even if no official calendar could be fetched.
            pred_rows = rows(ROOT / SPORTS[name][0])
            for row in pred_rows:
                kickoff = parse_dt(row.get("fecha"))
                old_enough = bool(kickoff and datetime.now(timezone.utc) - kickoff >= timedelta(hours=GRACE_HOURS))
                ticker = text(row.get("ticker"))
                historical = ticker.upper() in PLACEHOLDER_TICKERS
                alert = old_enough and not historical
                all_rows.append({
                    "deporte": name, "prediction_id": text(row.get("prediction_id")),
                    "match_id": text(row.get("match_id")), "fecha": text(row.get("fecha")),
                    "local": text(row.get("local")), "visita": text(row.get("visita")),
                    "bloque": text(row.get("bloque")), "ticker": ticker,
                    "estado_registrado": "", "estado_partido": "NO_VERIFICADO",
                    "estado_kalshi": "", "resultado_kalshi": "",
                    "motivo": "sin ticker (histórico)" if historical else "estado desconocido",
                    "clasificacion": "HISTORICO_SIN_CONTRATO" if historical else ("PENDIENTE_ACTIVA" if alert else "NO_VERIFICADO"),
                    "cuenta_alerta": "SI" if alert else "NO",
                    "horas_desde_inicio": round((datetime.now(timezone.utc)-kickoff).total_seconds()/3600, 2) if kickoff else "",
                    "resuelto_en": "",
                })
                total_pending += int(alert)

    # Always create the report, even with zero rows or if one sport's provider failed.
    with PENDING_REPORT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PENDING_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"INFORME DE PENDIENTES: {PENDING_REPORT.name} | filas={len(all_rows)} | alerta={total_pending} "
          f"| plazo_kalshi={GRACE_HOURS}h | plazo_busqueda_partido={MATCH_LOOKUP_GRACE_HOURS}h | errores={errors}")
    # The pipeline should still commit the diagnostic CSV. Fail the action only after
    # it has written the evidence, so the UI never falls back to a stale/missing report.
    if errors:
        raise SystemExit(2)
    if total_pending:
        raise SystemExit(3)
    # A red workflow means there are unresolved predictions on finished games; the CSV is already written.


if __name__ == "__main__":
    main()
