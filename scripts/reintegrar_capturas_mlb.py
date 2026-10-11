"""Restore specifically identified immutable MLB captures before settlement.

Rows are copied from the recovery ledger with their original captured_en timestamp.
This is idempotent and intentionally bypasses the pregame-only guard because these
are historical captures recovered from Git history, not new live predictions.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "predicciones_mlb.csv"
RECOVERED = ROOT / "predicciones_mlb_capturas_recuperadas.csv"
COLUMNS = [
    "prediction_id", "event_ticker", "ticker", "match_id", "fecha", "local", "visita",
    "categoria", "mercado", "line", "lado", "seleccion", "bloque", "probabilidad_modelo",
    "precio_captura", "probabilidad_implicita", "edge", "estado_modelo", "reglas",
    "capturado_en", "disponibilidad",
]
REQUIRED_IDS = {
    "KXMLBINNINGWIN-26OCT112000LADMIL-1-TIE|NO|SENCILLAS",
    "KXMLBGAME-26OCT151800TBCLE-TB|YES|SENCILLAS",
}
# The recovered Tampa-Cleveland row had been linked to the Oct 13 game even
# though its Kalshi ticker encodes Oct 15 18:00 ET. StatsAPI's matching fixture
# is gamePk 849818 (Oct 15 22:00 UTC, Cleveland home / Tampa Bay away).
OPERATIVE_METADATA_FIXES = {
    "KXMLBGAME-26OCT151800TBCLE-TB|YES|SENCILLAS": {
        "match_id": "849818",
        "fecha": "2026-10-15T22:00:00Z",
        "local": "Cleveland Guardians",
        "visita": "Tampa Bay Rays",
    },
}


def read_csv(path):
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def main():
    recovered, _ = read_csv(RECOVERED)
    existing, old_columns = read_csv(PRED)
    recovered_by_id = {str(row.get("prediction_id", "")).strip(): row for row in recovered}
    missing_recovery = REQUIRED_IDS - set(recovered_by_id)
    if missing_recovery:
        raise SystemExit(f"Faltan filas en archivo de recuperación: {sorted(missing_recovery)}")

    existing_ids = {str(row.get("prediction_id", "")).strip() for row in existing}
    added = []
    corrected = 0
    metadata_corrected = 0
    existing_by_id = {str(row.get("prediction_id", "")).strip(): row for row in existing}
    for pid in sorted(REQUIRED_IDS):
        if pid in existing_ids:
            live = existing_by_id[pid]
            original_capture = str(recovered_by_id[pid].get("capturado_en", "")).strip()
            if str(live.get("capturado_en", "")).strip() != original_capture:
                live["capturado_en"] = original_capture
                corrected += 1
                print(f"REINTEGRACIÓN MLB: corregida hora original de {pid}; capturado_en={original_capture}")
            else:
                print(f"REINTEGRACIÓN MLB: verificada {pid}; capturado_en original={original_capture}")
            for field, value in OPERATIVE_METADATA_FIXES.get(pid, {}).items():
                if str(live.get(field, "")).strip() != value:
                    print(f"REINTEGRACIÓN MLB: corregido enlace de partido de {pid}; {field}: {live.get(field, '')} -> {value}")
                    live[field] = value
                    metadata_corrected += 1
            continue
        row = dict(recovered_by_id[pid])
        if not row.get("capturado_en") or not row.get("ticker") or not row.get("fecha"):
            raise SystemExit(f"Fila recuperada incompleta: {pid}")
        row.update(OPERATIVE_METADATA_FIXES.get(pid, {}))
        added.append({key: row.get(key, "") for key in COLUMNS})
        print(f"REINTEGRACIÓN MLB: restaurada {pid}; capturado_en={row['capturado_en']}")

    if added or corrected or metadata_corrected:
        merged_columns = COLUMNS + [name for name in old_columns if name not in COLUMNS]
        tmp = PRED.with_name(PRED.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged_columns, extrasaction="ignore")
            writer.writeheader()
            for row in existing:
                writer.writerow({**{k: "" for k in merged_columns}, **row})
            for row in added:
                writer.writerow({**{k: "" for k in merged_columns}, **row})
        tmp.replace(PRED)
    print(f"REINTEGRACIÓN MLB: añadidas={len(added)}; horas_originales_corregidas={corrected}; campos_de_partido_corregidos={metadata_corrected}; filas_operativas={len(existing) + len(added)}")


if __name__ == "__main__":
    main()
