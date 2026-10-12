"""Shared Kalshi capture/settlement primitives: immutable pre-kickoff writes and official settlement."""
from __future__ import annotations
import csv, math, os, re, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import requests
from sports_results import settle_from_score

KALSHI_API_BASE = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2").rstrip("/")
TERMINAL_STATES = {"WIN", "LOSS", "ANULADA"}

def utc_now(): return datetime.now(timezone.utc)
def parse_datetime(value):
    if not value: return None
    try: dt = datetime.fromisoformat(str(value).replace("Z","+00:00"))
    except (TypeError, ValueError): return None
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
def extract_line(value):
    """Extract a line only when the market label explicitly states one."""
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(?i)(?:over|under|more than|less than|más de|menos de|handicap|spread|total|at least|at most)[^\d+-]*([+-]?\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", ".")
    match = re.search(r"(?<!\w)([+-]?\d+(?:[.,]\d+)?)\s*\+", text)
    if match:
        return match.group(1).replace(",", ".") + "+"
    match = re.search(r"(?<!\w)[+-]?\d+[.,]\d+(?!\w)", text)
    if match:
        return match.group(0).replace(",", ".")
    return ""


def valid_probability(value):
    try: p=float(value)
    except (TypeError,ValueError,OverflowError): return None
    return p if math.isfinite(p) and 0 <= p <= 1 else None

def probability_fraction(value):
    """Parse a probability or executable contract price conservatively.

    Accepts decimal values (0.57), percentage values ("57%") and dollar
    prices ("$0.57"). Missing, malformed, zero and unit-invalid prices return
    None so callers cannot manufacture an edge from unavailable quotes.
    """
    if value is None:
        return None
    raw = str(value).strip().replace(",", ".")
    if not raw or raw.casefold() in {"—", "-", "none", "nan", "n/a"}:
        return None
    is_percent = raw.endswith("%")
    is_cents = raw.endswith("¢")
    if is_percent or is_cents:
        raw = raw[:-1].strip()
    raw = raw.replace("$", "").strip()
    try:
        number = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    if is_percent or is_cents or number > 1:
        number /= 100
    return number if 0 < number < 1 else None


def calculate_edge(probability, executable_price):
    """Return model probability minus executable entry price, or None."""
    p = probability_fraction(probability)
    price = probability_fraction(executable_price)
    return round(p - price, 8) if p is not None and price is not None else None
def kalshi_get(path, params=None):
    error=None
    for attempt in range(4):
        try:
            r=requests.get(f"{KALSHI_API_BASE}/{path.lstrip('/')}",params=params or {},timeout=20)
            r.raise_for_status(); data=r.json()
            if not isinstance(data,dict): raise ValueError(f"JSON inesperado en {path}")
            return data
        except (requests.RequestException,ValueError) as exc:
            error=exc
            if attempt==3: break
            time.sleep(min(2**attempt,6))
    raise error
def iter_kalshi_pages(path,params,collection,max_pages=100):
    items=[]; cursor=None
    for _ in range(max_pages):
        query=dict(params)
        if cursor: query["cursor"]=cursor
        page=kalshi_get(path,query); items.extend(page.get(collection) or [])
        cursor=page.get("cursor")
        if not cursor: break
    return items
def sports_series(prefixes=()):
    rows=iter_kalshi_pages("series",{"limit":200,"category":"Sports"},"series",30)
    return [x for x in rows if not prefixes or str(x.get("ticker","")).startswith(tuple(prefixes))]
def events_for_series(series_tickers: Iterable[str],statuses=("open",)):
    events=[]; seen=set()
    for ticker in sorted(set(series_tickers)):
        for status in statuses:
            try: rows=iter_kalshi_pages("events",{"limit":200,"series_ticker":ticker,"status":status,"with_nested_markets":"true"},"events",50)
            except (requests.RequestException,ValueError,KeyError) as exc:
                print(f"AVISO Kalshi: no se pudo leer {ticker} ({status}): {exc}"); continue
            for event in rows:
                key=str(event.get("event_ticker") or "")
                if key and key not in seen: events.append(event); seen.add(key)
    return events
def verify_public_catalog():
    """Live smoke test against a real sports event and its complete nested market catalogue."""
    all_series = sports_series()
    if not all_series:
        raise RuntimeError("Kalshi no devolvió series deportivas.")
    selected = [x for x in all_series if str(x.get("ticker","")).startswith(("KXMLBGAME","KXMLBTOTAL","KXLALIGAGAME","KXUCLGAME"))]
    if not selected:
        raise RuntimeError("No hay series MLB/LaLiga/Champions en el catálogo deportivo de Kalshi.")
    for series in selected[:60]:
        events = events_for_series([series["ticker"]], statuses=("open",))
        for event in events:
            title = str(event.get("title") or "")
            if not re.search(r"\b(vs\.?|versus| at )\b", title, re.I):
                continue
            markets = [m for m in (event.get("markets") or []) if m.get("ticker")]
            if not markets:
                continue
            no_price = sum(
                not any(m.get(k) not in (None, "") for k in
                        ("yes_ask_dollars", "no_ask_dollars", "yes_bid_dollars", "no_bid_dollars"))
                for m in markets
            )
            sample = markets[0]
            return {
                "series_catalog_count": len(all_series),
                "sports_series_ticker": series.get("ticker"),
                "event_ticker": event.get("event_ticker"),
                "event_title": event.get("title"),
                "catalog_market_count": len(markets),
                "unpriced_market_count": no_price,
                "market_ticker": sample.get("ticker"),
                "market_title": sample.get("title") or sample.get("yes_sub_title"),
                "market_status": sample.get("status"),
                "yes_bid_dollars": sample.get("yes_bid_dollars"),
                "yes_ask_dollars": sample.get("yes_ask_dollars"),
                "no_bid_dollars": sample.get("no_bid_dollars"),
                "no_ask_dollars": sample.get("no_ask_dollars"),
            }
    raise RuntimeError("La API respondió, pero no se encontró un evento deportivo abierto con tickers reales en el catálogo.")

def read_rows(path):
    path=Path(path)
    if not path.exists() or not path.stat().st_size: return []
    with path.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))
def _write_header_if_missing(path,columns):
    if not path.exists() or not path.stat().st_size:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=columns).writeheader()
def ensure_csv_schema(path, columns):
    """Migrate a CSV header without changing any stored prediction values."""
    path = Path(path)
    columns = list(columns)
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()
        return columns
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_columns = list(reader.fieldnames or [])
        rows = list(reader)
    merged = columns + [name for name in old_columns if name not in columns]
    if old_columns != merged:
        for row in rows:
            if "mercado" in merged and not str(row.get("mercado", "")).strip():
                row["mercado"] = row.get("jugada") or ""
            if "line" in merged and not str(row.get("line", "")).strip():
                row["line"] = extract_line(row.get("mercado") or row.get("jugada") or "")
            if "seleccion" in merged and not str(row.get("seleccion", "")).strip():
                side = str(row.get("lado", "")).upper().strip()
                if side not in {"YES", "NO"}:
                    match = re.search(r"(YES|NO)[\"']?\s*\)?$", str(row.get("clave_interna", "")), re.I)
                    side = match.group(1).upper() if match else ""
                row["seleccion"] = side
            if "bloque" in merged and not str(row.get("bloque", "")).strip():
                category = str(row.get("categoria", ""))
                row["bloque"] = next((name for name in ("SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA") if category.startswith(name + " · ")), "")
            if "edge" in merged and not str(row.get("edge", "")).strip():
                p = valid_probability(row.get("probabilidad_calibrada", row.get("probabilidad_modelo")))
                try:
                    price = float(str(row.get("precio", row.get("precio_captura", ""))).replace("$", ""))
                except (TypeError, ValueError):
                    price = None
                if p is not None and price is not None and 0 <= price <= 1:
                    row["edge"] = f"{p - price:.6f}"
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    return merged


def append_immutable_predictions(path,columns,rows,*,probability_field,kickoff_field="fecha",capture_field="capturado_en",ticker_field="ticker",id_field="prediction_id"):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); columns=ensure_csv_schema(path,columns); existing=read_rows(path); ids={str(r.get(id_field,"")) for r in existing}; now=utc_now()
    accepted=[]; rejected=[]
    for source in rows:
        row=dict(source); pid=str(row.get(id_field,"")).strip(); ticker=str(row.get(ticker_field,"")).strip()
        p=valid_probability(row.get(probability_field)); kickoff=parse_datetime(row.get(kickoff_field)); captured=parse_datetime(row.get(capture_field)) or now
        reason=None
        if not pid: reason="sin prediction_id"
        elif pid in ids: reason="prediction_id ya guardado; no se sobrescribe"
        elif not ticker or ticker in {"—","-","None","nan"}: reason="sin ticker real de Kalshi"
        elif p is None: reason=f"probabilidad inválida en {probability_field}"
        elif "match_id" in row and not str(row.get("match_id","")).strip(): reason="sin partido asociado; no es una predicción de partido"
        elif kickoff is None: reason="hora de inicio ausente o inválida"
        elif captured>=kickoff or now>=kickoff: reason="partido iniciado antes de guardar"
        if reason: rejected.append((pid or "(sin id)",reason)); continue
        row[probability_field]=f"{p:.8f}"; row[capture_field]=captured.isoformat(); accepted.append(row); ids.add(pid)
    _write_header_if_missing(path,columns)
    if accepted:
        with path.open("a",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=columns,extrasaction="ignore").writerows(accepted)
    for pid,reason in rejected[:20]: print(f"CAPTURA DESCARTADA {pid}: {reason}")
    if len(rejected)>20: print(f"… y {len(rejected)-20} filas descartadas adicionales.")
    return {"accepted":len(accepted),"rejected":len(rejected)}
def _side_from_row(row,side_field):
    # Prefer explicit captured selection fields; legacy SENCILLAS rows may have
    # lost _lado when ranked, so infer only narrowly from unambiguous wording.
    for field in (side_field, "seleccion", "selection", "lado_kalshi"):
        side=str(row.get(field,"")).upper().strip() if field else ""
        if side in {"YES","NO"}: return side
    raw=row.get("clave_interna")
    parsed=None
    if raw:
        try:
            import ast
            parsed=ast.literal_eval(raw)
            if isinstance(parsed,tuple) and len(parsed)==2:
                side=str(parsed[1] or "").upper()
                if side in {"YES","NO"}: return side
        except (ValueError,SyntaxError,TypeError): pass
    # Conservative recovery for old captures where the ranked simple-bet row
    # dropped _lado. Never guess for spreads, totals, corners, or props.
    try:
        import ast
        parsed=parsed if parsed is not None else ast.literal_eval(str(raw or ""))
        contract=parsed[0]
        series=str(contract[0]).upper()
        outcome=str(contract[1]).lower()
        line=contract[2] if len(contract) > 2 else None
        pick=str(row.get("jugada") or "").casefold()
        if series.endswith("BTTS") and outcome in {"ambos","btts"}:
            if "sin marcar" in pick or "no marcan" in pick or "no marca" in pick:
                return "NO"
            if "ambos" in pick and "marcan" in pick:
                return "YES"
        # Recover a legacy team/total-corners side only when the saved Spanish
        # range and the exact Kalshi strike prove the binary side unambiguously.
        # Example: Kalshi ticker RBB6 means Real Betis 6+ corners; "0–5 corners"
        # is therefore NO. Do not infer from probability or from team names alone.
        if "CORNERS" in series and line is not None and "corner" in pick:
            try:
                strike = float(line)
                low_range = re.search(r"(?:^|\s)0\s*[–—-]\s*(\d+)", pick)
                high_range = re.search(r"(\d+)\s*[+]", pick)
                if low_range and abs(strike - (int(low_range.group(1)) + 1)) < 1e-9:
                    return "NO"
                if high_range and abs(strike - int(high_range.group(1))) < 1e-9:
                    return "YES"
            except (TypeError, ValueError):
                pass
        if series.endswith("GAME") and outcome in {"local","visita","empate"}:
            local=str(row.get("local") or "").casefold().strip()
            away=str(row.get("visita") or "").casefold().strip()
            if outcome=="empate":
                if "empate" in pick and not ("o " in pick or "local o" in pick or "visita o" in pick):
                    return "YES"
                if local and away and local in pick and away in pick and "empate" not in pick:
                    return "NO"
            else:
                team=local if outcome=="local" else away
                if team and ("gana " + team) in pick:
                    return "YES"
                if team and team in pick and ("local o empate" in pick or "visita o empate" in pick or "x2" in pick or "1x" in pick):
                    return "NO"
    except (ValueError,SyntaxError,TypeError,IndexError):
        pass
    return None
def _market_rows_for_series(series_tickers):
    found={}
    for series in sorted(set(series_tickers)):
        try: rows=iter_kalshi_pages("markets",{"limit":200,"series_ticker":series,"status":"settled"},"markets",100)
        except (requests.RequestException,ValueError,KeyError) as exc:
            print(f"AVISO liquidación Kalshi {series}: {exc}"); continue
        for market in rows:
            ticker=str(market.get("ticker") or "")
            if ticker: found[ticker]=market
    return found
def _trusted_terminal_result(row,result_columns,state_field="estado"):
    state=str(row.get(state_field,"")).upper()
    if state not in TERMINAL_STATES:
        return False
    marker=str(row.get("marcador",""))
    if marker.startswith("Marcador independiente ("):
        upper_marker=marker.upper()
        # Reject legacy records that accidentally used the full-time score for a half-time market.
        if "MERCADO KXLALIGA1H" in upper_marker or "MERCADO KXUCL1H" in upper_marker:
            if "1.ª PARTE" not in upper_marker:
                return False
        if "MERCADO KXLALIGA2H" in upper_marker or "MERCADO KXUCL2H" in upper_marker:
            if "2.ª PARTE" not in upper_marker:
                return False
        return "resultado_kalshi" not in result_columns
    if "resultado_kalshi" in result_columns:
        return str(row.get("resultado_kalshi","")).upper() in {"YES","NO","VOID"}
    return marker.startswith("Resultado oficial Kalshi:") or marker.startswith("Contrato anulado/cancelado por")

def liquidate_kalshi_ledger(predictions_path,results_path,result_columns,*,side_field="lado",ticker_field="ticker",id_field="prediction_id",state_field="estado",cancelled_match_ids=()):
    """Recover old predictions; use independent full-time scores where safe, Kalshi otherwise."""
    predictions_path=Path(predictions_path); results_path=Path(results_path)
    predictions=read_rows(predictions_path); results=read_rows(results_path)
    has_match_id=any("match_id" in r for r in predictions)
    # Settlement identity is the prediction ID, not ticker/probability validity.
    # Historical rows without a Kalshi ticker remain in the prediction file and are
    # classified by the reconciler; they must not disappear from audit counts.
    valid_prediction_ids={
        str(row.get(id_field,"")) for row in predictions
        if str(row.get(id_field,"")).strip()
    }
    terminal={str(r.get(id_field,"")) for r in results
              if str(r.get(id_field,"")) in valid_prediction_ids
              and _trusted_terminal_result(r,result_columns,state_field)}
    now=utc_now()
    cancelled_match_ids={str(value) for value in (cancelled_match_ids or ())}
    pending=[]
    new=[]
    skipped=0
    for row in predictions:
        pid=str(row.get(id_field,""))
        if pid not in valid_prediction_ids or pid in terminal:
            continue
        match_id=str(row.get("match_id","")).strip()
        if match_id and match_id in cancelled_match_ids:
            out={id_field:pid,state_field:"ANULADA",
                 "marcador":"Contrato anulado/cancelado por fuente oficial: partido aplazado o cancelado",
                 "resuelto_en":utc_now().isoformat()}
            if "resultado_kalshi" in result_columns:
                out["resultado_kalshi"]="VOID"
            new.append(out)
            terminal.add(pid)
            continue
        kickoff=parse_datetime(row.get("fecha"))
        # Future fixtures do not need a settlement lookup yet. Past fixtures are
        # always retried, regardless of age, so missed schedules are recovered.
        if kickoff is None or kickoff > now:
            continue
        # A score-based result may resolve a captured prediction without an executable ticker.
        pending.append(row)
    series={str(r[ticker_field]).split("-",1)[0] for r in pending
            if str(r.get(ticker_field,"")).strip().upper() not in {"","—","-","NONE","NAN"}}
    # For small backlogs, query exact tickers directly; scanning every settled market in a series is needlessly slow.
    markets=_market_rows_for_series(series) if len(pending) > 50 else {}
    exact_market_cache={}
    for row in pending:
        pid=str(row.get(id_field,"")); ticker=str(row.get(ticker_field,"")).strip()
        # Prefer an independent scoreboard for markets whose outcome follows safely
        # from the full-time score. Corners, halves, first goal, cards and props fall back to Kalshi.
        external = settle_from_score(row)
        if external and external.get("state") in {"WIN", "LOSS"}:
            out={id_field:pid,state_field:external["state"],"marcador":external["reason"],
                 "resuelto_en":utc_now().isoformat()}
            new.append(out)
            terminal.add(pid)
            continue
        if ticker.upper() in {"","—","-","NONE","NAN"}:
            skipped += 1
            continue
        market=markets.get(ticker)
        if market is None:
            # Always retry exact tickers, including old backlogs. Cache each ticker
            # so predictions in separate blocks do not cause duplicate API calls.
            if ticker not in exact_market_cache:
                found = None
                # Route 1: exact market endpoint.
                try:
                    payload = kalshi_get(f"markets/{ticker}")
                    found = payload.get("market") or next(
                        (item for item in (payload.get("markets") or [])
                         if str(item.get("ticker") or "") == ticker), None
                    )
                except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
                    pass
                # Route 2: market-list endpoint filtered by the exact ticker. Some
                # older/delisted contracts are absent from the direct endpoint while
                # still discoverable through the list endpoint.
                if not found:
                    try:
                        payload = kalshi_get("markets", {"tickers": ticker, "limit": 1000})
                        found = next(
                            (item for item in (payload.get("markets") or [])
                             if str(item.get("ticker") or "") == ticker), None
                        )
                    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
                        pass
                exact_market_cache[ticker] = found
            market=exact_market_cache[ticker]
            if market is None:
                skipped += 1
                continue
        result=str(market.get("result") or "").lower().strip(); status=str(market.get("status") or "").lower().strip()
        side=_side_from_row(row,side_field)
        if result in {"yes","no"} and side in {"YES","NO"}:
            state="WIN" if result.upper()==side else "LOSS"; reason=f"Resultado oficial Kalshi: {result.upper()}"
        elif result in {"void","canceled","cancelled"} or status in {"canceled","cancelled","void"}:
            state="ANULADA"; reason=f"Contrato anulado/cancelado por Kalshi (status={status or 'sin estado'}, result={result or 'sin resultado'})"
        else: skipped+=1; continue
        out={id_field:pid,state_field:state,"marcador":reason,"resuelto_en":utc_now().isoformat()}
        if "resultado_kalshi" in result_columns: out["resultado_kalshi"]=result.upper() if result else "VOID"
        new.append(out); terminal.add(pid)
    # The liquidator is the sole writer of result files: keep only the latest terminal
    # state per prediction. Legacy NO_EVALUABLE rows are removed and retried until Kalshi settles.
    latest={}
    for row in results + new:
        state=str(row.get(state_field,"")).upper()
        if not _trusted_terminal_result(row,result_columns,state_field):
            continue
        pid=str(row.get(id_field,""))
        if pid not in valid_prediction_ids:
            continue
        previous=latest.get(pid)
        if previous is None or str(row.get("resuelto_en","")) >= str(previous.get("resuelto_en","")):
            latest[pid]=row
    results_path.parent.mkdir(parents=True,exist_ok=True)
    with results_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=result_columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(latest.values())
    summary={"predicciones_leidas":len(predictions),"pendientes_revisadas":len(pending),"resultados_nuevos":len(new),
        "win":sum(r["estado"]=="WIN" for r in new),"loss":sum(r["estado"]=="LOSS" for r in new),
        "anuladas":sum(r["estado"]=="ANULADA" for r in new),"sin_resolver_aun":skipped}
    print("LIQUIDACIÓN HÍBRIDA (marcador independiente + Kalshi):",summary); return summary
def read_csv_rows_resilient(path):
    """Repair mixed-schema CSV history; quarantine unrecoverable raw rows before rewriting."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = f.readlines()
    if not lines:
        return []
    try:
        header = next(csv.reader([lines[0]], strict=True))
    except (csv.Error, StopIteration) as exc:
        raise ValueError(f"{path.name}: cabecera CSV ilegible: {exc}") from exc
    if not header:
        return []
    legacy_football = [
        "prediction_id", "match_id", "fecha", "local", "visita", "categoria", "jugada",
        "probabilidad_base", "probabilidad_calibrada", "calibracion", "precio",
        "disponibilidad", "ticker", "clave_interna", "capturado_en",
    ]
    valid, bad = [], []
    migrated = False
    for line_no, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            continue
        try:
            values = next(csv.reader([raw], strict=True))
        except (csv.Error, StopIteration) as exc:
            bad.append((line_no, raw.rstrip("\r\n"), f"CSV inválido: {exc}"))
            continue
        if len(values) == len(header):
            valid.append(dict(zip(header, values)))
            continue
        # Historical football rows had 15 columns before market/line/side/block/edge were added.
        if (path.name in {"predicciones_laliga.csv", "predicciones_champions.csv"}
                and len(values) == len(legacy_football) and len(header) >= 20):
            row = dict(zip(legacy_football, values))
            row.update({column: "" for column in header if column not in row})
            row["mercado"] = row.get("jugada", "")
            row["line"] = extract_line(row.get("jugada", ""))
            category = str(row.get("categoria", ""))
            row["bloque"] = next((name for name in ("SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA")
                                  if category.startswith(name + " · ")), "SENCILLAS")
            valid.append(row)
            migrated = True
            continue
        # Recover unquoted commas in the free-text marker of 4-column settlement ledgers.
        if len(header) == 4 and header[-1] == "resuelto_en" and len(values) > 4:
            timestamp = values[-1]
            if re.match(r"^\d{4}-\d{2}-\d{2}T", timestamp):
                valid.append(dict(zip(header, [values[0], values[1], ",".join(values[2:-1]), timestamp])))
                migrated = True
                continue
        bad.append((line_no, raw.rstrip("\r\n"),
                    f"{len(values)} campos; se esperaban {len(header)}"))
    if bad:
        quarantine = path.with_suffix(path.suffix + ".malformed_rows.csv")
        seen = set()
        if quarantine.exists():
            try:
                with quarantine.open("r", encoding="utf-8-sig", newline="") as f:
                    seen = {row.get("raw_line", "") for row in csv.DictReader(f)}
            except (OSError, csv.Error):
                seen = set()
        new_bad = [item for item in bad if item[1] not in seen]
        needs_header = not quarantine.exists() or quarantine.stat().st_size == 0
        if new_bad:
            with quarantine.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["source", "line_number", "reason", "raw_line"])
                if needs_header:
                    writer.writeheader()
                for line_no, raw, reason in new_bad:
                    writer.writerow({"source": path.name, "line_number": line_no,
                                     "reason": reason, "raw_line": raw})
    if bad or migrated:
        # The sidecar is durable before the primary CSV is rewritten; no malformed source row is lost.
        with path.open("w", encoding="utf-8", newline="") as repaired:
            writer = csv.writer(repaired)
            writer.writerow(header)
            for row in valid:
                writer.writerow([row.get(column, "") for column in header])
        if bad:
            print(f"CSV HISTORIAL REPARADO: {path.name}: {len(bad)} fila(s) preservada(s) en "
                  f"{quarantine.name}; {len(valid)} filas válidas conservadas.")
        elif migrated:
            print(f"CSV HISTORIAL MIGRADO: {path.name}: {len(valid)} filas conservadas en el esquema actual.")
    return valid



def _base_probability(row):
    """Calibrated model probability before the conservative market anchor."""
    for key in ("probabilidad_calibrada", "Probabilidad calibrada",
                "Prob. recalibrada", "Probabilidad modelo",
                "Probabilidad del modelo", "Prob. modelo", "probabilidad_modelo",
                "probabilidad_modelo_original", "Probabilidad base", "_p"):
        raw = row.get(key)
        p = valid_probability(raw)
        if p is not None:
            return p
        text_value = str(raw or "").strip()
        match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text_value)
        if match:
            try:
                number = float(match.group(0).replace(",", "."))
                if "%" in text_value or number > 1:
                    number /= 100
                p = valid_probability(number)
                if p is not None:
                    return p
            except (TypeError, ValueError, OverflowError):
                pass
    return None


def _market_has_chronological_evidence(category, min_samples=None):
    """Require explicit out-of-time evidence before increasing a market's model weight.

    Ordinary descriptive Brier summaries are deliberately insufficient: the report
    must mark the market as chronologically validated, meet the configurable sample
    minimum, and beat the contemporaneous Kalshi price on Brier score.
    """
    report = Path(os.getenv("BACKTEST_MERCADO_PATH", "backtest_mlb_resumen.csv"))
    try:
        minimum = int(min_samples if min_samples is not None else os.getenv("MUESTRA_MINIMA_BACKTEST", "100"))
    except (TypeError, ValueError, OverflowError):
        minimum = 100
    if minimum < 1:
        minimum = 100
    target = re.sub(r"\s+", " ", str(category or "").strip()).casefold()
    if not target or not report.exists():
        return False
    try:
        with report.open(newline="", encoding="utf-8-sig") as f:
            rows = csv.DictReader(f)
            for row in rows:
                name = re.sub(r"\s+", " ", str(row.get("tipo_mercado", row.get("mercado", ""))).strip()).casefold()
                if name != target:
                    continue
                try:
                    n = int(float(row.get("n_resueltos", row.get("n", 0))))
                    model_brier = float(row.get("brier_modelo", "nan"))
                    market_brier = float(row.get("brier_precio_kalshi", "nan"))
                except (TypeError, ValueError, OverflowError):
                    return False
                temporal = str(row.get("validacion_cronologica", "")).strip().casefold() in {"1", "true", "yes", "sí", "si", "validado"}
                return temporal and n >= minimum and math.isfinite(model_brier) and math.isfinite(market_brier) and model_brier < market_brier
    except (OSError, csv.Error):
        return False
    return False


def _market_model_weight(category=None, ticker=None):
    """Default to a 5% model weight; increase only with explicit chronological evidence."""
    try:
        base_weight = float(os.getenv("PESO_MODELO_MERCADO", "0.05"))
    except (TypeError, ValueError, OverflowError):
        base_weight = 0.05
    if not math.isfinite(base_weight):
        base_weight = 0.05
    base_weight = min(0.25, max(0.0, base_weight))

    market = re.sub(
        r"^(SENCILLAS|ALTA_PROBABILIDAD|A_TENER_EN_CUENTA)\s*[·|:-]\s*",
        "", str(category or "").strip(), flags=re.I,
    )
    if _market_has_chronological_evidence(market):
        try:
            validated_weight = float(os.getenv("PESO_MODELO_MERCADO_VALIDADO", "0.25"))
        except (TypeError, ValueError, OverflowError):
            validated_weight = 0.25
        if not math.isfinite(validated_weight):
            validated_weight = 0.25
        return min(0.25, max(base_weight, validated_weight))
    return base_weight


def anchor_probability_to_market(model_probability, market_price, weight=None, category=None, ticker=None):
    """Blend a calibrated model probability with the selected contract's executable ask.

    The price is the implied probability of the purchased YES/NO side. Missing or
    invalid prices intentionally produce None rather than a model-only 'final'.
    The model weight is 5% by default; only backtested LaLiga categories can use 25%.
    """
    model_p = probability_fraction(model_probability)
    price = probability_fraction(market_price)
    if model_p is None or price is None:
        return None
    if weight is None:
        weight = _market_model_weight(category, ticker)
    try:
        weight = float(weight)
    except (TypeError, ValueError, OverflowError):
        weight = _market_model_weight(category, ticker)
    if not math.isfinite(weight):
        weight = _market_model_weight(category, ticker)
    weight = min(0.25, max(0.0, weight))
    return min(0.995, max(0.005, price + weight * (model_p - price)))


def _probability(row):
    # Ranking/edge decisions must use the final market-anchored probability.
    final = valid_probability(row.get("probabilidad_final", row.get("Probabilidad final")))
    if final is not None:
        return final
    return _base_probability(row)


def _anchor_market_row(source):
    row = dict(source)
    # Recompute after calibration changes the model estimate; never reuse a stale final.
    model_p = _base_probability(row)
    price = _candidate_price(row)
    category = row.get("Tipo", row.get("Categoría resumen", row.get("categoria", row.get("bloque", ""))))
    ticker = row.get("Ticker", row.get("ticker", ""))
    final = anchor_probability_to_market(model_p, price, category=category, ticker=ticker)
    if model_p is not None:
        row.setdefault("probabilidad_modelo_original", model_p)
        row["Probabilidad del modelo"] = f"{model_p:.1%}"
    if price is not None:
        row["Precio implícito Kalshi"] = price
    if final is not None:
        row["probabilidad_final"] = final
        row["Probabilidad final"] = f"{final:.1%}"
        row["Peso modelo aplicado"] = _market_model_weight(category, ticker)
    else:
        row["probabilidad_final"] = ""
        row["Probabilidad final"] = "Sin precio válido"
    return row
def _family(ticker,label=""):
    series=str(ticker or "").split("-",1)[0].upper()
    text=f"{series} {label}".upper()
    if "CORNER" in text: return "corners"
    if any(x in text for x in ("BTTS","FTTS","FIRSTGOAL","TEAMTOTAL","TTOTAL","TOTAL","1H","2H")): return "goals_related"
    if any(x in text for x in ("SPREAD","GAME","WIN")): return "result"
    return series or "other"

def rank_market_candidates(rows,min_probability=0.65,high_limit=5,value_limit=10):
    """Conservative shared gate for combo/value lists; never fill weak candidates."""
    min_edge = float(os.getenv("MERCADOS_EDGE_MIN_NETO", "0.05"))
    cost_buffer = max(0.0, float(os.getenv("MERCADOS_BUFFER_COSTES", "0.01")))
    combo_price_max = float(os.getenv("COMBO_PRECIO_MAX", "0.90"))
    value_price_max = float(os.getenv("VALOR_PRECIO_MAX", "0.88"))
    max_auto_edge = float(os.getenv("MERCADOS_EDGE_MAX_AUTO", "0.20"))
    candidates = []
    for source in rows or []:
        row = _anchor_market_row(source)
        ticker = str(row.get("Ticker") or row.get("ticker") or "").strip()
        p = _probability(row)
        price = _candidate_price(row)
        if not ticker or ticker in {"—", "-", "None", "nan"} or p is None or price is None:
            continue
        if p <= 0.01 or p >= 0.99:
            continue
        # Recompute edge from calibrated probability and executable price, not stale CSV edge.
        edge = p - price - cost_buffer
        if edge < min_edge or edge > max_auto_edge or _is_trivial_single(row):
            continue
        item = dict(row)
        item["_prob"], item["_price"], item["_edge"] = p, price, edge
        item["_family"] = _family(ticker, row.get("Apuesta") or row.get("Jugada sencilla") or "")
        item["Ventaja (edge)"] = round(edge, 6)
        if "edge" in item:
            item["edge"] = round(edge, 6)
        candidates.append(item)
    high, families = [], set()
    for item in sorted(candidates, key=lambda x: (x["_prob"], x["_edge"]), reverse=True):
        if item["_price"] > combo_price_max or item["_prob"] < min_probability or item["_family"] in families:
            continue
        high.append(item)
        families.add(item["_family"])
        if len(high) >= high_limit:
            break
    value, line_values, labels, seen_tickers = [], {}, set(), set()
    for item in sorted(candidates, key=lambda x: (x["_edge"], x["_prob"]), reverse=True):
        if item["_price"] > value_price_max:
            continue
        ticker = str(item.get("Ticker") or item.get("ticker") or "")
        if ticker in seen_tickers:
            continue
        label = str(item.get("Apuesta") or item.get("Jugada sencilla") or "")
        line = re.search(r"\b\d+(?:\.\d+)?\b", label)
        family = item["_family"]
        if line:
            number = float(line.group(0))
            prior = line_values.setdefault(family, [])
            if any(abs(number-old) <= 1.0 for old in prior):
                continue
            prior.append(number)
        else:
            signature = (family, label.casefold())
            if signature in labels:
                continue
            labels.add(signature)
        seen_tickers.add(ticker)
        value.append(item)
        if len(value) >= value_limit:
            break
    def clean(items):
        return [{k: v for k, v in item.items() if k not in {"_prob", "_family", "_edge", "_price"}} for item in items]
    return clean(high), clean(value)


def _candidate_price(row):
    for key in ("Precio de compra", "Precio ahora", "Precio Kalshi", "precio_captura", "precio", "price"):
        raw = row.get(key)
        if raw in (None, "", "—", "-"):
            continue
        try:
            value = float(re.search(r"[-+]?\d+(?:[.,]\d+)?", str(raw)).group(0).replace(",", "."))
            if "%" in str(raw) or value > 1:
                value /= 100
            if math.isfinite(value) and 0 < value < 1:
                return value
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
    return None



def evaluate_market_recommendation(row, *, predictions_path=None, results_path=None, min_samples=None):
    """Fail-closed recommendation gate: valid ask, enough settlements, and Brier better than ask."""
    ticker = str(row.get("Ticker") or row.get("ticker") or "").strip()
    if not ticker or ticker in {"—", "-", "None", "nan"}:
        return {"eligible": False, "reason": "no hay ticker real de Kalshi", "n": 0}
    series = ticker.split("-", 1)[0].upper()
    root = Path(__file__).resolve().parent
    if predictions_path is None or results_path is None:
        if series.startswith("KXMLB"):
            predictions_path = predictions_path or root / "predicciones_mlb.csv"
            results_path = results_path or root / "resultados_mlb.csv"
        elif series.startswith("KXLALIGA"):
            predictions_path = predictions_path or root / "predicciones_laliga.csv"
            results_path = results_path or root / "resultados_laliga.csv"
        elif series.startswith(("KXUCL", "KXCHAMP")):
            predictions_path = predictions_path or root / "predicciones_champions.csv"
            results_path = results_path or root / "resultados_champions.csv"
        elif series.startswith(("KXNFL", "KXNFL")):
            predictions_path = predictions_path or root / "predicciones_nfl.csv"
            results_path = results_path or root / "resultados_nfl.csv"
        else:
            return {"eligible": False, "reason": "no hay historial de liquidaciones para este mercado", "n": 0}

    ask_raw = None
    for key in ("Precio de compra", "Precio de compra (ask)", "ask", "ask_price", "precio_ask"):
        if row.get(key) not in (None, "", "—", "-"):
            ask_raw = row.get(key)
            break
    if ask_raw is None:
        side = str(row.get("_lado") or row.get("Lado Kalshi") or row.get("lado") or row.get("seleccion") or "").upper()
        ask_keys = ("yes_ask_dollars", "yes_ask") if side == "YES" else ("no_ask_dollars", "no_ask")
        ask_raw = next((row.get(k) for k in ask_keys if row.get(k) not in (None, "", "—", "-")), None)
    ask = probability_fraction(ask_raw)
    if ask is None:
        return {"eligible": False, "reason": "precio de compra (ask) no válido", "n": 0}

    try:
        minimum = int(min_samples if min_samples is not None else os.getenv("MUESTRA_MINIMA_RECOMENDACION", "100"))
    except (TypeError, ValueError, OverflowError):
        minimum = 100
    minimum = max(1, minimum)
    predictions = read_rows(predictions_path)
    result_rows = read_rows(results_path)
    latest_results = {}
    for result in sorted(result_rows, key=lambda r: str(r.get("resuelto_en", ""))):
        pid = str(result.get("prediction_id", "")).strip()
        if pid:
            latest_results[pid] = str(result.get("estado", result.get("estado_resultado", ""))).upper().strip()

    def clean_category(value):
        value = re.sub(r"^(SENCILLAS|ALTA_PROBABILIDAD|A_TENER_EN_CUENTA)\\s*[·|:-]\\s*", "", str(value or "").strip(), flags=re.I)
        value = re.sub(r"\\s*[·|:-]\\s*(YES|NO)$", "", value, flags=re.I)
        return re.sub(r"\\s+", " ", value).casefold()

    target_category = clean_category(row.get("Tipo", row.get("Categoría resumen", row.get("categoria", row.get("bloque", "")))))
    observations = []
    for old in predictions:
        old_ticker = str(old.get("ticker", old.get("Ticker", ""))).strip().upper()
        if not old_ticker or old_ticker.split("-", 1)[0] != series:
            continue
        if target_category and clean_category(old.get("categoria", old.get("Tipo", ""))) != target_category:
            continue
        state = latest_results.get(str(old.get("prediction_id", "")).strip(), "")
        if state not in {"WIN", "LOSS"}:
            continue
        p = None
        for key in ("probabilidad_calibrada", "probabilidad_modelo", "probabilidad_base"):
            p = probability_fraction(old.get(key))
            if p is not None:
                break
        price = None
        for key in ("precio_captura", "precio", "Precio de compra", "precio_kalshi"):
            price = probability_fraction(old.get(key))
            if price is not None:
                break
        if p is None or price is None:
            continue
        y = 1.0 if state == "WIN" else 0.0
        observations.append(((p-y)**2, (price-y)**2))

    n = len(observations)
    if n < minimum:
        return {"eligible": False, "reason": f"muestra insuficiente: {n}/{minimum} resultados WIN/LOSS del mismo mercado", "n": n}
    brier_model = sum(x[0] for x in observations) / n
    brier_market = sum(x[1] for x in observations) / n
    if not brier_model < brier_market:
        return {
            "eligible": False,
            "reason": f"el modelo no supera al precio de Kalshi en Brier ({brier_model:.3f} vs {brier_market:.3f})",
            "n": n, "brier_model": brier_model, "brier_market": brier_market,
        }
    return {"eligible": True, "reason": f"Brier mejor que Kalshi con {n} resultados", "n": n,
            "brier_model": brier_model, "brier_market": brier_market}

def _candidate_edge(row, price):
    # Recompute using calibrated probability and executable price; reserve 1 pp for costs by default.
    p = _probability(row)
    if p is None or price is None:
        return None
    try:
        buffer = max(0.0, float(os.getenv("MERCADOS_BUFFER_COSTES", "0.01")))
    except (TypeError, ValueError, OverflowError):
        buffer = 0.01
    return p - price - buffer


def _single_family(row):
    label = " ".join(str(row.get(k) or "") for k in (
        "Categoría resumen", "categoria", "Tipo", "Apuesta sencilla", "Apuesta", "mercado", "Ticker", "ticker"
    )).casefold()
    ticker = str(row.get("Ticker") or row.get("ticker") or "").upper()
    if any(x in label for x in ("corner", "córner")):
        return "corners"
    if any(x in label for x in ("strikeout", "ponche", "pitcher", "home run", "batter", "bateador", "hits allowed", "bases total", "carreras impulsadas")):
        return "jugadores:" + (str(row.get("categoria") or row.get("Categoría resumen") or "jugadores").casefold())
    if any(x in label for x in ("ambos marcan", "both teams", "btts")):
        return "ambos_marcan"
    if any(x in label for x in ("primer gol", "first goal", "first scorer", "primeras 5 entradas", "first 5", "f5")):
        return "periodo_especial:" + (str(row.get("categoria") or row.get("Categoría resumen") or ticker).casefold())
    if any(x in label for x in ("total exact", "total runs", "total carreras", "goles totales", "goals total", "over ", "under ", "más de ", "menos de ", "team total", "total goles")):
        if "team total" in label or "total por equipo" in label or "goles por equipo" in label or "carreras por equipo" in label:
            return "totales_equipo"
        return "totales_partido"
    if any(x in label for x in ("moneyline", "ganador", "resultado", "spread", "hándicap", "handicap", "run line", "gana ", "win")):
        return "resultado"
    category = str(row.get("categoria") or row.get("Categoría resumen") or "").casefold()
    if category:
        return "categoria:" + category
    return "serie:" + (ticker.split("-", 1)[0] or "otros")


def _is_trivial_single(row):
    label = " ".join(str(row.get(k) or "") for k in (
        "Apuesta sencilla", "Apuesta", "mercado", "line", "seleccion"
    )).casefold()
    category = " ".join(str(row.get(k) or "") for k in ("categoria", "Categoría resumen")).casefold()
    total_market = any(word in label + " " + category for word in (
        "gol", "goal", "carrera", "run", "corner", "córner", "total"
    ))
    line = str(row.get("line") or "").strip().replace(",", ".")
    if total_market and line in {"0.5", "0.50"}:
        return True
    if re.search(r"(?:más de|mas de|over|at least|more than|or more)\s*0[.,]5|0[.,]5\s*(?:\+|or more|or higher)", label):
        return total_market
    return False


def single_candidate_rejection_reasons(rows, *, price_max=None, min_edge=None):
    """Count why raw candidates do not pass the simple-bet quality gate."""
    if price_max is None:
        price_max = float(os.getenv("SENCILLAS_PRECIO_MAX", "0.88"))
    if min_edge is None:
        min_edge = float(os.getenv("SENCILLAS_EDGE_MIN", "0.05"))
    counts = {"sin ticker real": 0, "probabilidad inválida": 0, "sin precio Kalshi": 0,
              "precio por encima del máximo": 0, "ventaja insuficiente": 0,
              "ventaja extrema requiere revisión": 0, "línea trivial": 0,
              "candidata que sí cumple": 0}
    for source in rows or []:
        row = _anchor_market_row(source)
        ticker = str(row.get("Ticker") or row.get("ticker") or "").strip()
        p = _probability(row)
        price = _candidate_price(row)
        edge = _candidate_edge(row, price)
        if not ticker or ticker in {"—", "-", "None", "nan"}:
            counts["sin ticker real"] += 1
        elif p is None:
            counts["probabilidad inválida"] += 1
        elif price is None:
            counts["sin precio Kalshi"] += 1
        elif price > price_max:
            counts["precio por encima del máximo"] += 1
        elif edge is None or edge < min_edge:
            counts["ventaja insuficiente"] += 1
        elif edge > float(os.getenv("SENCILLAS_EDGE_REVISAR", "0.20")):
            counts["ventaja extrema requiere revisión"] += 1
        elif p <= 0.01 or p >= 0.99:
            counts["probabilidad inválida"] += 1
        elif _is_trivial_single(row):
            counts["línea trivial"] += 1
        else:
            counts["candidata que sí cumple"] += 1
    return {key: value for key, value in counts.items() if value}


def rank_single_candidates(rows, *, price_max=None, min_edge=None, review_edge=None, limit=10):
    """Select simple bets by model edge, avoiding trivial prices and correlated duplicates.

    Defaults are environment-configurable: SENCILLAS_PRECIO_MAX=0.88,
    SENCILLAS_EDGE_MIN=0.05 and SENCILLAS_EDGE_REVISAR=0.20.
    """
    if price_max is None:
        price_max = float(os.getenv("SENCILLAS_PRECIO_MAX", "0.88"))
    if min_edge is None:
        min_edge = float(os.getenv("SENCILLAS_EDGE_MIN", "0.05"))
    if review_edge is None:
        review_edge = float(os.getenv("SENCILLAS_EDGE_REVISAR", "0.20"))
    accepted = []
    for source in rows or []:
        row = dict(source)
        nested = row.get("_row")
        if isinstance(nested, dict):
            combined = dict(nested)
            combined.update({k: v for k, v in row.items() if k != "_row"})
            combined["_row"] = nested
            row = combined
        row = _anchor_market_row(row)
        ticker = str(row.get("Ticker") or row.get("ticker") or "").strip()
        p = _probability(row)
        price = _candidate_price(row)
        edge = _candidate_edge(row, price)
        if not ticker or ticker in {"—", "-", "None", "nan"} or p is None or price is None or edge is None:
            continue
        if (price > price_max or edge < min_edge or _is_trivial_single(row)
                or p <= 0.01 or p >= 0.99 or edge > review_edge):
            # Extreme apparent edge requires review for stale quotes/model-data errors.
            continue
        row["_single_price"] = price
        row["_single_edge"] = edge
        row["_single_p"] = p
        row["_single_family"] = _single_family(row)
        row["Revisión"] = "REVISAR" if edge > review_edge else ""
        accepted.append(row)
    selected = []
    families = set()
    used_teams = set()
    for row in sorted(accepted, key=lambda x: (x["_single_edge"], x["_single_p"]), reverse=True):
        family = row["_single_family"]
        if family in families:
            continue
        label = " ".join(str(row.get(k) or "") for k in ("Apuesta sencilla", "Apuesta", "mercado"))
        # Keep the list diversified: no more than two selections tied to the same named team.
        teams = re.findall(r"(?<![\w])([A-Z][A-Za-zÀ-ÿ.'-]+(?:\s+[A-Z][A-Za-zÀ-ÿ.'-]+){0,2})", str(row.get("Apuesta sencilla") or row.get("Apuesta") or ""))
        # Ignore generic bet phrases: "Más de" is not a team and must not suppress totals.
        teams = [team for team in teams if team.casefold() not in {
            "más", "mas", "menos", "más de", "mas de", "menos de", "over", "under", "ambos", "ambos equipos marcan",
            "más de goles", "menos de goles",
        }]
        if len(used_teams) >= 2 and teams and all(team.casefold() in used_teams for team in teams):
            continue
        selected.append(row)
        families.add(family)
        used_teams.update(team.casefold() for team in teams)
        if len(selected) >= limit:
            break
    for row in selected:
        row["Ventaja (edge)"] = round(row["_single_edge"], 6)
        model_p = _base_probability(row)
        row["Probabilidad base"] = model_p if model_p is not None else ""
        row["Precio de compra"] = f"{round(row['_single_price'] * 100)}¢"
        row["Probabilidad del modelo"] = f"{model_p:.1%}" if model_p is not None else "—"
        row["Probabilidad final"] = f"{row['_single_p']:.1%}"
        row["probabilidad_final"] = row["_single_p"]
        for key in ("_single_price", "_single_edge", "_single_p", "_single_family", "_p"):
            row.pop(key, None)
    return selected
