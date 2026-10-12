"""Chronological, paired Brier backtest: model vs captured Kalshi price by LaLiga market."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MIN_SAMPLES_FOR_WEIGHT_INCREASE = 30
BASE_MODEL_WEIGHT = 0.05
VALIDATED_MODEL_WEIGHT = 0.25


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def probability(value):
    text = str(value or "").strip().replace("$", "")
    if not text or text.casefold() in {"—", "-", "none", "nan", "n/a"}:
        return None
    is_percent = text.endswith("%")
    is_cents = text.endswith("¢")
    if is_percent or is_cents:
        text = text[:-1].strip()
    try:
        number = float(text.replace(",", "."))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    if is_percent or is_cents or number > 1:
        number /= 100
    return number if 0 < number < 1 else None


def category_name(value):
    text = str(value or "").strip()
    text = re.sub(r"^(SENCILLAS|ALTA_PROBABILIDAD|A_TENER_EN_CUENTA)\s*[·|:-]\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).casefold()


def parse_dt(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def brier(values):
    return sum(values) / len(values) if values else None


def build_backtest(prediction_rows, result_rows):
    # Use the most recently recorded settlement per prediction_id.
    latest_result = {}
    for row in result_rows:
        pid = str(row.get("prediction_id", "")).strip()
        if not pid:
            continue
        prior = latest_result.get(pid)
        if prior is None or str(row.get("resuelto_en", "")) > str(prior.get("resuelto_en", "")):
            latest_result[pid] = row

    # Prediction IDs are immutable; if old files contain duplicates, keep the earliest
    # valid pre-kickoff capture so the same forecast cannot be counted more than once.
    unique_predictions = {}
    for row in prediction_rows:
        pid = str(row.get("prediction_id", "")).strip()
        if not pid or pid in unique_predictions:
            continue
        unique_predictions[pid] = row

    grouped = defaultdict(list)
    paired = []
    excluded = {"no_result": 0, "not_pre_kickoff": 0, "no_ticker": 0, "invalid_probability": 0, "invalid_price": 0}
    for pid, row in unique_predictions.items():
        result = latest_result.get(pid)
        state = str((result or {}).get("estado", "")).strip().upper()
        if state not in {"WIN", "LOSS"}:
            excluded["no_result"] += 1
            continue
        kickoff = parse_dt(row.get("fecha"))
        captured = parse_dt(row.get("capturado_en"))
        if kickoff is None or captured is None or captured >= kickoff:
            excluded["not_pre_kickoff"] += 1
            continue
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or ticker in {"—", "-", "NONE", "NAN"}:
            excluded["no_ticker"] += 1
            continue
        model_p = probability(row.get("probabilidad_calibrada"))
        if model_p is None:
            model_p = probability(row.get("probabilidad_base"))
        price_p = probability(row.get("precio"))
        if model_p is None:
            excluded["invalid_probability"] += 1
            continue
        if price_p is None:
            excluded["invalid_price"] += 1
            continue
        item = {
            "prediction_id": pid,
            "category": category_name(row.get("categoria")),
            "model_p": model_p,
            "price_p": price_p,
            "outcome": 1.0 if state == "WIN" else 0.0,
        }
        paired.append(item)
        grouped[item["category"]].append(item)

    def summarize(rows):
        if not rows:
            return {"n": 0, "brier_model": None, "brier_price": None,
                    "brier_blend_5pct": None, "brier_blend_25pct": None,
                    "model_beats_price": False, "blend_25pct_beats_price": False,
                    "peso_modelo_recomendado": BASE_MODEL_WEIGHT}
        y = [r["outcome"] for r in rows]
        model = [(r["model_p"] - r["outcome"]) ** 2 for r in rows]
        price = [(r["price_p"] - r["outcome"]) ** 2 for r in rows]
        blend5 = [((r["price_p"] + BASE_MODEL_WEIGHT * (r["model_p"] - r["price_p"])) - r["outcome"]) ** 2 for r in rows]
        blend25 = [((r["price_p"] + VALIDATED_MODEL_WEIGHT * (r["model_p"] - r["price_p"])) - r["outcome"]) ** 2 for r in rows]
        model_brier, price_brier = brier(model), brier(price)
        blend25_brier = brier(blend25)
        model_beats = model_brier < price_brier
        blend_beats = blend25_brier < price_brier
        enough = len(rows) >= MIN_SAMPLES_FOR_WEIGHT_INCREASE
        recommended = VALIDATED_MODEL_WEIGHT if enough and model_beats and blend_beats else BASE_MODEL_WEIGHT
        return {
            "n": len(rows),
            "brier_model": round(model_brier, 6),
            "brier_price": round(price_brier, 6),
            "brier_blend_5pct": round(brier(blend5), 6),
            "brier_blend_25pct": round(blend25_brier, 6),
            "model_beats_price": model_beats,
            "blend_25pct_beats_price": blend_beats,
            "peso_modelo_recomendado": recommended,
        }

    categories = {key: summarize(rows) for key, rows in sorted(grouped.items())}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Solo predicciones con captura anterior al inicio, ticker real, precio capturado válido y resultado WIN/LOSS; Brier del modelo y precio se compara sobre exactamente las mismas filas.",
        "minimum_samples_for_weight_increase": MIN_SAMPLES_FOR_WEIGHT_INCREASE,
        "base_model_weight": BASE_MODEL_WEIGHT,
        "validated_model_weight": VALIDATED_MODEL_WEIGHT,
        "total": summarize(paired),
        "paired_rows": len(paired),
        "excluded_rows": excluded,
        "categories": categories,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "predicciones_laliga.csv")
    parser.add_argument("--results", type=Path, default=ROOT / "resultados_laliga.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "backtest_probabilidades_mercado.json")
    args = parser.parse_args()
    report = build_backtest(read_csv(args.predictions), read_csv(args.results))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"BACKTEST MERCADO: paired_rows={report['paired_rows']} output={args.output}")


if __name__ == "__main__":
    main()
