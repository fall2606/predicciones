"""Shared Kalshi capture/settlement primitives: immutable pre-kickoff writes and official settlement."""
from __future__ import annotations
import csv, math, os, re, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import requests

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
    side=str(row.get(side_field,"")).upper().strip() if side_field else ""
    if side in {"YES","NO"}: return side
    raw=row.get("clave_interna")
    if raw:
        try:
            import ast
            parsed=ast.literal_eval(raw)
            if isinstance(parsed,tuple) and len(parsed)==2:
                side=str(parsed[1]).upper()
                if side in {"YES","NO"}: return side
        except (ValueError,SyntaxError,TypeError): pass
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
    if "resultado_kalshi" in result_columns:
        return str(row.get("resultado_kalshi","")).upper() in {"YES","NO","VOID"}
    marker=str(row.get("marcador",""))
    return marker.startswith("Resultado oficial Kalshi:") or marker.startswith("Contrato anulado/cancelado por Kalshi")

def liquidate_kalshi_ledger(predictions_path,results_path,result_columns,*,side_field="lado",ticker_field="ticker",id_field="prediction_id",state_field="estado"):
    """Recover every old prediction; trust Kalshi's final YES/NO/void outcome."""
    predictions_path=Path(predictions_path); results_path=Path(results_path)
    predictions=read_rows(predictions_path); results=read_rows(results_path)
    has_match_id=any("match_id" in r for r in predictions)
    valid_prediction_ids={
        str(row.get(id_field,"")) for row in predictions
        if str(row.get(id_field,"")).strip()
        and str(row.get(ticker_field,"")).strip() not in {"","—","-","None","nan"}
        and valid_probability(row.get("probabilidad_calibrada",row.get("probabilidad_modelo"))) is not None
        and (not has_match_id or str(row.get("match_id","")).strip())
    }
    terminal={str(r.get(id_field,"")) for r in results
              if str(r.get(id_field,"")) in valid_prediction_ids
              and _trusted_terminal_result(r,result_columns,state_field)}
    now=utc_now()
    pending=[]
    for row in predictions:
        pid=str(row.get(id_field,""))
        if pid not in valid_prediction_ids or pid in terminal:
            continue
        kickoff=parse_datetime(row.get("fecha"))
        # Future fixtures do not need a settlement lookup yet. Past fixtures are
        # always retried, regardless of age, so missed schedules are recovered.
        if kickoff is None or kickoff > now:
            continue
        pending.append(row)
    series={str(r[ticker_field]).split("-",1)[0] for r in pending if r.get(ticker_field)}
    # For small backlogs, query exact tickers directly; scanning every settled market in a series is needlessly slow.
    markets=_market_rows_for_series(series) if len(pending) > 50 else {}; new=[]; skipped=0
    for row in pending:
        pid=str(row.get(id_field,"")); ticker=str(row.get(ticker_field,"")).strip(); market=markets.get(ticker)
        if market is None:
            # Avoid thousands of per-ticker calls during backlog recovery. Once the
            # series catalogue is mostly resolved, retry individual missing tickers.
            if len(pending) > 50:
                skipped += 1
                continue
            try:
                payload=kalshi_get(f"markets/{ticker}"); market=payload.get("market") or (payload.get("markets") or [{}])[0]
            except (requests.RequestException,ValueError,KeyError,IndexError,TypeError): skipped+=1; continue
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
    print("LIQUIDACIÓN KALSHI:",summary); return summary
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



def _probability(row):
    p=valid_probability(row.get("_p"))
    if p is not None: return p
    raw=str(row.get("Probabilidad del modelo",row.get("Probabilidad modelo",row.get("Prob. modelo",row.get("probabilidad_modelo","")))))
    m=re.search(r"[-+]?\d+(?:[.,]\d+)?",raw)
    if not m: return None
    n=float(m.group(0).replace(",","."))
    if "%" in raw: n/=100
    return valid_probability(n)
def _family(ticker,label=""):
    series=str(ticker or "").split("-",1)[0].upper()
    text=f"{series} {label}".upper()
    if "CORNER" in text: return "corners"
    if any(x in text for x in ("BTTS","FTTS","FIRSTGOAL","TEAMTOTAL","TTOTAL","TOTAL","1H","2H")): return "goals_related"
    if any(x in text for x in ("SPREAD","GAME","WIN")): return "result"
    return series or "other"

def rank_market_candidates(rows,min_probability=0.65,high_limit=5,value_limit=10):
    candidates=[]
    for row in rows or []:
        ticker=str(row.get("Ticker") or row.get("ticker") or "").strip()
        p=_probability(row)
        if not ticker or ticker in {"—","-","None","nan"} or p is None: continue
        price=_candidate_price(row)
        item=dict(row); item["_prob"]=p; item["_price"]=price
        item["_family"]=_family(ticker,row.get("Apuesta") or row.get("Jugada sencilla") or "")
        try:
            edge=float(row.get("Ventaja (edge)"))
            if not math.isfinite(edge): edge=None
        except (TypeError,ValueError,OverflowError): edge=None
        item["_edge"]=edge; candidates.append(item)
    high=[]; families=set()
    for x in sorted(candidates,key=lambda r:r["_prob"],reverse=True):
        # Avoid near-certain prices in combo suggestions, without changing the value block.
        if x["_price"] is None or x["_price"] > float(os.getenv("COMBO_PRECIO_MAX", "0.90")): continue
        if x["_prob"]<min_probability or x["_family"] in families: continue
        high.append(x); families.add(x["_family"])
        if len(high)>=high_limit: break
    value=[]; line_values={}; labels=set(); seen_tickers=set()
    for x in sorted((r for r in candidates if r["_edge"] is not None),key=lambda r:(r["_edge"],r["_prob"]),reverse=True):
        ticker=str(x.get("Ticker") or x.get("ticker") or "")
        if ticker in seen_tickers: continue
        seen_tickers.add(ticker)
        label=str(x.get("Apuesta") or x.get("Jugada sencilla") or "")
        line=re.search(r"\b\d+(?:\.\d+)?\b",label)
        family=x["_family"]
        if line:
            number=float(line.group(0))
            prior=line_values.setdefault(family,[])
            if any(abs(number-old)<=1.0 for old in prior): continue
            prior.append(number)
        else:
            signature=(family,label.casefold())
            if signature in labels: continue
            labels.add(signature)
        value.append(x)
        if len(value)>=value_limit: break
    def clean(items):
        return [{k:v for k,v in x.items() if k not in {"_prob","_family","_edge","_price"}} for x in items]
    return clean(high),clean(value)


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


def _candidate_edge(row, price):
    for key in ("Ventaja (edge)", "edge"):
        try:
            value = float(row.get(key))
            if math.isfinite(value):
                # Some legacy CSVs store edge in percentage points.
                return value / 100 if abs(value) > 1 else value
        except (TypeError, ValueError, OverflowError):
            pass
    p = _probability(row)
    return p - price if p is not None and price is not None else None


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
              "línea trivial": 0, "candidata que sí cumple": 0}
    for source in rows or []:
        row = dict(source)
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
        ticker = str(row.get("Ticker") or row.get("ticker") or "").strip()
        p = _probability(row)
        price = _candidate_price(row)
        edge = _candidate_edge(row, price)
        if not ticker or ticker in {"—", "-", "None", "nan"} or p is None or price is None or edge is None:
            continue
        if price > price_max or edge < min_edge or _is_trivial_single(row):
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
        row["Probabilidad base"] = row["_single_p"]
        row["Precio de compra"] = f"{row['_single_price']:.0%}"
        row["Probabilidad del modelo"] = f"{row['_single_p']:.1%}"
        for key in ("_single_price", "_single_edge", "_single_p", "_single_family", "_p", "_lado"):
            row.pop(key, None)
    return selected
