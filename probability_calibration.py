"""Leakage-safe probability calibration from prior settled prediction snapshots."""
from __future__ import annotations
import csv
import math
import re
from pathlib import Path

TERMINAL = {"WIN", "LOSS"}


def _read(path):
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _prob(value):
    try:
        p = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return p if math.isfinite(p) and 0 <= p <= 1 else None


def _category(value):
    text = str(value or "").strip()
    text = re.sub(r"^(SENCILLAS|ALTA_PROBABILIDAD|A_TENER_EN_CUENTA)\s*[·|:-]\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).casefold()


def calibrate_probability(probability, category, predictions_path, results_path,
                          min_samples=30, neighborhood=0.10, prior_strength=20, exclude_prediction_ids=None):
    """Return (calibrated_probability, explanation); fallback is always original p.

    Only settled WIN/LOSS outcomes from earlier captured forecasts are used.
    Same-category observations within +/- 10 percentage points are shrunk toward
    the original forecast; no adjustment is learned before 30 comparable cases.
    """
    p = _prob(probability)
    if p is None:
        return probability, "probabilidad inválida; sin calibración"
    predictions = _read(predictions_path)
    results = {str(r.get("prediction_id", "")).strip(): str(r.get("estado", "")).upper().strip()
               for r in _read(results_path)}
    target = _category(category)
    observations = []
    excluded = {str(x).strip() for x in (exclude_prediction_ids or ())}
    for row in predictions:
        prediction_id = str(row.get("prediction_id", "")).strip()
        if prediction_id in excluded:
            continue
        state = results.get(prediction_id, "")
        if state not in TERMINAL or _category(row.get("categoria")) != target:
            continue
        old_p = _prob(row.get("probabilidad_modelo", row.get("probabilidad_calibrada", row.get("probabilidad_base"))))
        if old_p is not None and abs(old_p - p) <= neighborhood:
            observations.append((old_p, state == "WIN"))
    n = len(observations)
    if n < min_samples:
        return p, f"sin ajuste: {n}/{min_samples} casos comparables"
    wins = sum(1 for _, won in observations if won)
    adjusted = (wins + prior_strength * p) / (n + prior_strength)
    adjusted = min(0.995, max(0.005, adjusted))
    return adjusted, f"calibrada con {n} casos comparables"
