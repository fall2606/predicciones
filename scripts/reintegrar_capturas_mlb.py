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
    for pid in sorted(REQUIRED_IDS):
        if pid in existing_ids:
            print(f"REINTEGRACIÓN MLB: ya existe {pid}; se conserva la fila operativa.")
            continue
        row = recovered_by_id[pid]
        if not row.get("capturado_en") or not row.get("ticker") or not row.get("fecha"):
            raise SystemExit(f"Fila recuperada incompleta: {pid}")
        added.append({key: row.get(key, "") for key in COLUMNS})
        print(f"REINTEGRACIÓN MLB: restaurada {pid}; capturado_en={row['capturado_en']}")

    if added:
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
    print(f"REINTEGRACIÓN MLB: añadidas={len(added)}; filas_operativas={len(existing) + len(added)}")


if __name__ == "__main__":
    main()
