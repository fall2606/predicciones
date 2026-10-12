"""Auditoría reproducible de resultados, probabilidades y calibración.

No pretende ser un backtest fuera de muestra: identifica los límites de los datos
archivados y no recomienda apuestas.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ("WIN", "LOSS", "ANULADA", "VOID", "CANCELLED", "CANCELED", "PUSH")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_probability(value):
    text = str(value or "").strip()
    if not text:
        return None
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    try:
        p = float(text.replace(",", "."))
    except (ValueError, TypeError, OverflowError):
        return None
    if percent:
        p /= 100
    return p if math.isfinite(p) and 0 <= p <= 1 else None


def score_rows(predictions, results, probability_fields):
    """Score by immutable prediction_id; last result by resolved timestamp wins."""
    result_counts = Counter(str(r.get("prediction_id", "")).strip() for r in results)
    ordered = sorted(results, key=lambda r: str(r.get("resuelto_en", "")))
    by_id = {}
    for row in ordered:
        key = str(row.get("prediction_id", "")).strip()
        if key:
            by_id[key] = row
    scored, status_counts = [], Counter()
    missing_id = duplicate_prediction_ids = 0
    seen = set()
    for pred in predictions:
        pid = str(pred.get("prediction_id", "")).strip()
        if not pid:
            missing_id += 1
            continue
        if pid in seen:
            duplicate_prediction_ids += 1
        seen.add(pid)
        result = by_id.get(pid)
        status = str((result or {}).get("estado", "SIN_RESULTADO")).strip().upper()
        status_counts[status or "SIN_RESULTADO"] += 1
        if status not in {"WIN", "LOSS"}:
            continue
        p = next((v for field in probability_fields if (v := parse_probability(pred.get(field))) is not None), None)
        if p is None:
            continue
        scored.append({"p": p, "y": 1.0 if status == "WIN" else 0.0, "status": status,
                       "capturado_en": pred.get("capturado_en", ""), "fecha": pred.get("fecha", ""),
                       "prediction_id": pid})
    n = len(scored)
    if n:
        rate = sum(r["y"] for r in scored) / n
        brier = sum((r["p"] - r["y"]) ** 2 for r in scored) / n
        eps = 1e-15
        logloss = -sum(r["y"] * math.log(max(eps, min(1-eps, r["p"]))) +
                       (1-r["y"]) * math.log(max(eps, min(1-eps, 1-r["p"]))) for r in scored) / n
        bins = []
        for index in range(10):
            low = index / 10
            group = [r for r in scored if min(int(r["p"] * 10), 9) == index]
            if group:
                bins.append({"intervalo": f"{low:.0%}-{low+.1:.0%}", "n": len(group),
                             "probabilidad_media": sum(r["p"] for r in group)/len(group),
                             "tasa_acierto": sum(r["y"] for r in group)/len(group)})
        metrics = {"n_evaluable": n, "tasa_acierto": rate, "probabilidad_media": sum(r["p"] for r in scored)/n,
                   "brier": brier, "log_loss": logloss,
                   "brier_referencia_muestra": rate * (1-rate),
                   "calibracion_intervalos": bins}
    else:
        metrics = {"n_evaluable": 0, "tasa_acierto": None, "probabilidad_media": None,
                   "brier": None, "log_loss": None, "brier_referencia_muestra": None,
                   "calibracion_intervalos": []}
    terminal_ids = {pid for pid, row in by_id.items() if str(row.get("estado", "")).upper() in RESULTADOS}
    prediction_ids = {str(p.get("prediction_id", "")).strip() for p in predictions if str(p.get("prediction_id", "")).strip()}
    return {"capturas": len(predictions), "filas_resultados": len(results),
            "ids_resultado_terminal_en_archivo": len(terminal_ids),
            "capturas_con_resultado_terminal": len(prediction_ids & terminal_ids),
            "sin_resultado_terminal": sum(1 for p in predictions if str(p.get("prediction_id", "")).strip() not in terminal_ids),
            "prediction_id_vacios": missing_id, "prediction_id_duplicados_en_capturas": duplicate_prediction_ids,
            "ids_resultado_duplicados": sum(1 for n in result_counts.values() if n > 1),
            "estados": dict(status_counts), **metrics,
            "limitacion": "Diagnóstico descriptivo; no es backtest cronológico fuera de muestra."}


def summarize_reconciliation(rows):
    """Summarize the canonical reconciliation export without hiding alert rows."""
    counts = Counter()
    for row in rows:
        key = (
            str(row.get("deporte", "") or "SIN_DEPORTE").strip(),
            str(row.get("clasificacion", "") or "SIN_CLASIFICACION").strip(),
            str(row.get("cuenta_alerta", "") or "SIN_MARCA_ALERTA").strip().upper(),
            str(row.get("motivo", "") or "SIN_MOTIVO").strip(),
        )
        counts[key] += 1
    return [
        {"deporte": k[0], "clasificacion": k[1], "cuenta_alerta": k[2], "motivo": k[3], "filas": n}
        for k, n in sorted(counts.items())
    ]


def audit_file(name, predictions_file, results_file, probability_fields):
    predictions = read_csv(ROOT / predictions_file)
    results = read_csv(ROOT / results_file)
    return {"deporte": name, "archivo_predicciones": predictions_file, "archivo_resultados": results_file,
            **score_rows(predictions, results, probability_fields)}


def main():
    audits = [
        audit_file("LaLiga", "predicciones_laliga.csv", "resultados_laliga.csv", ("probabilidad_calibrada", "probabilidad_base")),
        audit_file("Champions League", "predicciones_champions.csv", "resultados_champions.csv", ("probabilidad_calibrada", "probabilidad_base")),
        audit_file("MLB", "predicciones_mlb.csv", "resultados_mlb.csv", ("probabilidad_calibrada", "probabilidad_modelo")),
        audit_file("NFL", "predicciones_nfl.csv", "resultados_nfl.csv", ("probabilidad_calibrada", "probabilidad_modelo")),
    ]
    (ROOT / "auditoria_calidad_actual.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    reconciliation = summarize_reconciliation(read_csv(ROOT / "reconciliacion_pendientes.csv"))
    with (ROOT / "auditoria_reconciliacion_actual.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["deporte", "clasificacion", "cuenta_alerta", "motivo", "filas"])
        writer.writeheader()
        writer.writerows(reconciliation)
    fields = ["deporte", "capturas", "filas_resultados", "capturas_con_resultado_terminal", "sin_resultado_terminal",
              "prediction_id_vacios", "prediction_id_duplicados_en_capturas", "ids_resultado_duplicados", "n_evaluable",
              "tasa_acierto", "probabilidad_media", "brier", "log_loss", "brier_referencia_muestra", "limitacion"]
    with (ROOT / "auditoria_calidad_actual.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in audits:
            writer.writerow({k: (f"{row[k]:.6f}" if isinstance(row.get(k), float) else row.get(k, "")) for k in fields})
    print("AUDITORÍA DESCRIPTIVA DE CALIDAD (no equivale a backtest fuera de muestra)")
    for row in audits:
        print(f"{row['deporte']}: capturas={row['capturas']}; terminales enlazados={row['capturas_con_resultado_terminal']}; "
              f"sin resultado terminal={row['sin_resultado_terminal']}; evaluables={row['n_evaluable']}; "
              f"Brier={row['brier'] if row['brier'] is not None else 'N/D'}; "
              f"log-loss={row['log_loss'] if row['log_loss'] is not None else 'N/D'}; "
              f"IDs duplicados en capturas={row['prediction_id_duplicados_en_capturas']}; "
              f"IDs duplicados en resultados={row['ids_resultado_duplicados']}")
    print("RECONCILIACIÓN: desglose de todas las filas del archivo canónico")
    for row in reconciliation:
        print(f"{row['deporte']} · {row['clasificacion']} · alerta={row['cuenta_alerta']} · filas={row['filas']} · motivo={row['motivo']}")
    print("Informes generados: auditoria_calidad_actual.csv, auditoria_calidad_actual.json y auditoria_reconciliacion_actual.csv")


if __name__ == "__main__":
    main()
