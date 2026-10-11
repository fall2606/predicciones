"""One-time, idempotent repair for prediction CSVs appended under mixed schema versions."""
import ast
import csv
import math
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_FOOTBALL = [
    "prediction_id", "match_id", "fecha", "local", "visita", "categoria", "jugada",
    "probabilidad_base", "probabilidad_calibrada", "calibracion", "precio",
    "disponibilidad", "ticker", "clave_interna", "capturado_en",
]
CURRENT_FOOTBALL = [
    "prediction_id", "match_id", "fecha", "local", "visita", "categoria", "jugada",
    "probabilidad_base", "probabilidad_calibrada", "calibracion", "precio",
    "disponibilidad", "ticker", "mercado", "line", "seleccion", "bloque", "edge",
    "clave_interna", "capturado_en",
]
LEGACY_MLB = [
    "prediction_id", "event_ticker", "ticker", "match_id", "fecha", "local", "visita",
    "categoria", "mercado", "lado", "probabilidad_modelo", "precio_captura",
    "probabilidad_implicita", "edge", "estado_modelo", "reglas", "capturado_en",
]
CURRENT_MLB = [
    "prediction_id", "event_ticker", "ticker", "match_id", "fecha", "local", "visita",
    "categoria", "mercado", "line", "lado", "seleccion", "bloque",
    "probabilidad_modelo", "precio_captura", "probabilidad_implicita", "edge",
    "estado_modelo", "reglas", "capturado_en", "disponibilidad",
]


def _line_from(row, allow_line_correction=False):
    text = str(row.get("mercado") or row.get("jugada") or "")
    if allow_line_correction:
        plus = re.search(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*\+", text)
        if plus:
            return plus.group(1).replace(",", ".") + "+"
    if str(row.get("line", "")).strip():
        return str(row["line"]).strip()
    raw = str(row.get("clave_interna", "") or "")
    try:
        parsed = ast.literal_eval(raw)
        key = parsed[0] if isinstance(parsed, tuple) and len(parsed) == 2 else parsed
        if isinstance(key, tuple) and len(key) >= 3 and key[2] is not None:
            return str(key[2])
    except (ValueError, SyntaxError, TypeError):
        pass
    match = re.search(r"(?i)(?:over|under|más de|menos de|gana por|handicap|spread|total)[^\d+-]*([+-]?\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", ".")
    match = re.search(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*\+", text)
    if match:
        return match.group(1).replace(",", ".") + "+"
    return ""


def _metadata(row, allow_line_correction=False):
    if not str(row.get("mercado", "")).strip():
        row["mercado"] = row.get("jugada") or row.get("ticker") or ""
    row["line"] = _line_from(row, allow_line_correction=allow_line_correction)
    side = str(row.get("seleccion") or row.get("lado") or "").upper().strip()
    if side not in {"YES", "NO"}:
        try:
            parsed = ast.literal_eval(str(row.get("clave_interna", "")))
            side = str(parsed[1]).upper() if isinstance(parsed, tuple) and len(parsed) == 2 else ""
        except (ValueError, SyntaxError, TypeError):
            side = ""
    row["seleccion"] = side if side in {"YES", "NO"} else ""
    if ("estado_modelo" in row or "precio_captura" in row) and not str(row.get("disponibilidad", "")).strip():
        price = str(row.get("precio_captura", "")).strip()
        row["disponibilidad"] = "Disponible en Kalshi" if price else "Sin precio en Kalshi"
    category = str(row.get("categoria", "") or "")
    block = str(row.get("bloque", "") or "").strip()
    if block not in {"SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA"}:
        detected = next((name for name in ("SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA")
                         if category.startswith(name + " · ")), "")
        # Old non-game contracts (expansion/futures) are archived but are not a match block.
        block = detected or ("SENCILLAS" if str(row.get("match_id", "")).strip() else "")
    row["bloque"] = block
    if not str(row.get("edge", "")).strip():
        p_raw = row.get("probabilidad_calibrada", row.get("probabilidad_modelo", ""))
        price_raw = row.get("precio", row.get("precio_captura", ""))
        try:
            p = float(p_raw)
            price_text = str(price_raw).strip().replace("$", "")
            price = float(price_text.rstrip("%")) / (100 if price_text.endswith("%") else 1)
            if math.isfinite(p) and 0 <= p <= 1 and math.isfinite(price) and 0 <= price <= 1:
                row["edge"] = f"{p-price:.6f}"
        except (TypeError, ValueError, OverflowError):
            pass
    return row


def repair_file(path, current_columns, legacy_columns):
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return {"file": path.name, "rows": 0, "repaired": False, "schemas": {}}
    with path.open(newline="", encoding="utf-8-sig") as f:
        raw_rows = list(csv.reader(f))
    if not raw_rows:
        return {"file": path.name, "rows": 0, "repaired": False, "schemas": {}}
    header = raw_rows[0]
    data = raw_rows[1:]
    repaired_rows, schemas = [], {}
    metadata_modified = False
    quarantined_rows = []
    removed_invalid_pregame = 0
    removed_over_limit_pregame = 0
    for line_no, values in enumerate(data, start=2):
        if not values or (len(values) == 1 and not values[0]):
            continue
        if len(values) == len(current_columns):
            source_columns = current_columns
        elif len(values) == len(header):
            source_columns = header
        elif len(values) == len(legacy_columns):
            source_columns = legacy_columns
        else:
            quarantined_rows.append((line_no, values,
                f"fila con {len(values)} campos; cabecera={len(header)}, esquema_actual={len(current_columns)}, legado={len(legacy_columns)}"))
            continue
        row = dict(zip(source_columns, values))
        # Two MLB series families were incorrectly attached to a single game in the
        # latest pregame capture. Remove only while that game's kickoff is still ahead.
        if path.name == "predicciones_mlb.csv" and str(row.get("match_id", "")).strip():
            ticker = str(row.get("ticker", "")).upper()
            try:
                kickoff = datetime.fromisoformat(str(row.get("fecha", "")).replace("Z", "+00:00"))
                if kickoff.tzinfo is None:
                    kickoff = kickoff.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                kickoff = None
            if kickoff is not None and datetime.now(timezone.utc) < kickoff.astimezone(timezone.utc) and any(
                bad in ticker for bad in ("KXMLBSERIESGAMES", "KXMLBINNINGTOTAL")
            ):
                removed_invalid_pregame += 1
                continue
        allow_line_correction = False
        if str(row.get("match_id", "")).strip():
            try:
                kickoff_for_line = datetime.fromisoformat(str(row.get("fecha", "")).replace("Z", "+00:00"))
                if kickoff_for_line.tzinfo is None:
                    kickoff_for_line = kickoff_for_line.replace(tzinfo=timezone.utc)
                allow_line_correction = datetime.now(timezone.utc) < kickoff_for_line.astimezone(timezone.utc)
            except (TypeError, ValueError):
                allow_line_correction = False
        before = dict(row)
        row = _metadata(row, allow_line_correction=allow_line_correction)
        if row != before:
            metadata_modified = True
        repaired_rows.append({column: row.get(column, "") for column in current_columns})
        schemas[str(len(values))] = schemas.get(str(len(values)), 0) + 1
    if quarantined_rows:
        quarantine = path.with_suffix(path.suffix + ".malformed_rows.csv")
        seen = set()
        if quarantine.exists():
            try:
                with quarantine.open("r", newline="", encoding="utf-8-sig") as f:
                    seen = {row.get("raw_fields", "") for row in csv.DictReader(f)}
            except (OSError, csv.Error):
                seen = set()
        new_bad = [item for item in quarantined_rows if repr(item[1]) not in seen]
        if new_bad:
            needs_header = not quarantine.exists() or quarantine.stat().st_size == 0
            with quarantine.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["source", "line_number", "reason", "raw_fields"])
                if needs_header:
                    writer.writeheader()
                for line_no, values, reason in new_bad:
                    writer.writerow({"source": path.name, "line_number": line_no,
                                     "reason": reason, "raw_fields": repr(values)})
        print(f"CSV REPARACIÓN: {path.name}: {len(quarantined_rows)} fila(s) irregular(es) preservada(s) en {quarantine.name}.")

    if path.name == "predicciones_mlb.csv":
        now = datetime.now(timezone.utc)
        groups = {}
        for index, row in enumerate(repaired_rows):
            match_id = str(row.get("match_id", "")).strip()
            if not match_id:
                continue
            try:
                kickoff = datetime.fromisoformat(str(row.get("fecha", "")).replace("Z", "+00:00"))
                if kickoff.tzinfo is None:
                    kickoff = kickoff.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
            if now >= kickoff.astimezone(timezone.utc):
                continue
            block = str(row.get("bloque", ""))
            if block in {"SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA"}:
                groups.setdefault((match_id, block), []).append((index, row))
        limits = {"SENCILLAS": 5, "ALTA_PROBABILIDAD": 5, "A_TENER_EN_CUENTA": 10}
        drop = set()
        for (_, block), items in groups.items():
            limit = limits[block]
            if len(items) <= limit:
                continue
            def score(item):
                row = item[1]
                field = "edge" if block == "A_TENER_EN_CUENTA" else "probabilidad_modelo"
                try:
                    value = float(row.get(field, ""))
                    if not math.isfinite(value):
                        value = -math.inf
                except (TypeError, ValueError, OverflowError):
                    value = -math.inf
                try:
                    probability = float(row.get("probabilidad_modelo", ""))
                    if not math.isfinite(probability):
                        probability = -math.inf
                except (TypeError, ValueError, OverflowError):
                    probability = -math.inf
                return (value, probability, str(row.get("capturado_en", "")))
            keep = {index for index, _ in sorted(items, key=score, reverse=True)[:limit]}
            for index, _ in items:
                if index not in keep:
                    drop.add(index)
        if drop:
            removed_over_limit_pregame = len(drop)
            repaired_rows = [row for index, row in enumerate(repaired_rows) if index not in drop]

    changed = header != current_columns or metadata_modified or removed_invalid_pregame > 0 or removed_over_limit_pregame > 0 or bool(quarantined_rows) or any(
        len(values) != len(header) for values in data if values
    )
    if changed:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=current_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(repaired_rows)
        tmp.replace(path)
    return {"file": path.name, "rows": len(repaired_rows), "repaired": changed or removed_invalid_pregame > 0, "schemas": schemas, "removed_invalid_pregame": removed_invalid_pregame, "removed_over_limit_pregame": removed_over_limit_pregame, "quarantined_rows": len(quarantined_rows)}


def main():
    specs = [
        ("predicciones_laliga.csv", CURRENT_FOOTBALL, LEGACY_FOOTBALL),
        ("predicciones_champions.csv", CURRENT_FOOTBALL, LEGACY_FOOTBALL),
        ("predicciones_mlb.csv", CURRENT_MLB, LEGACY_MLB),
    ]
    for filename, current, legacy in specs:
        print("REPARACIÓN CSV:", repair_file(ROOT / filename, current, legacy))


if __name__ == "__main__":
    main()
