"""One-time, idempotent repair for prediction CSVs appended under mixed schema versions."""
import ast
import csv
import math
import re
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
    "estado_modelo", "reglas", "capturado_en",
]


def _line_from(row):
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
    text = str(row.get("mercado") or row.get("jugada") or "")
    match = re.search(r"(?i)(?:over|under|más de|menos de|gana por|handicap|spread|total)[^\d+-]*([+-]?\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", ".")
    match = re.search(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*\+", text)
    if match:
        return match.group(1).replace(",", ".") + "+"
    return ""


def _metadata(row):
    if not str(row.get("mercado", "")).strip():
        row["mercado"] = row.get("jugada") or row.get("ticker") or ""
    row["line"] = _line_from(row)
    side = str(row.get("seleccion") or row.get("lado") or "").upper().strip()
    if side not in {"YES", "NO"}:
        try:
            parsed = ast.literal_eval(str(row.get("clave_interna", "")))
            side = str(parsed[1]).upper() if isinstance(parsed, tuple) and len(parsed) == 2 else ""
        except (ValueError, SyntaxError, TypeError):
            side = ""
    row["seleccion"] = side if side in {"YES", "NO"} else ""
    category = str(row.get("categoria", "") or "")
    block = str(row.get("bloque", "") or "").strip()
    if block not in {"SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA"}:
        block = next((name for name in ("SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA")
                      if category.startswith(name + " · ")), "SENCILLAS")
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
            raise ValueError(
                f"{path.name}:{line_no}: fila con {len(values)} campos; "
                f"cabecera={len(header)}, esquema_actual={len(current_columns)}, "
                f"legado={len(legacy_columns)}. Se detiene para no perder datos."
            )
        row = dict(zip(source_columns, values))
        before = dict(row)
        row = _metadata(row)
        if row != before:
            metadata_modified = True
        repaired_rows.append({column: row.get(column, "") for column in current_columns})
        schemas[str(len(values))] = schemas.get(str(len(values)), 0) + 1
    changed = header != current_columns or metadata_modified or any(
        len(values) != len(header) for values in data if values
    )
    if changed:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=current_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(repaired_rows)
        tmp.replace(path)
    return {"file": path.name, "rows": len(repaired_rows), "repaired": changed, "schemas": schemas}


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
