"""Walk-forward NFL backtest; proxy lines are explicitly not presented as historical Kalshi quotes."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import nfl_engine as nfl

SUMMARY = ROOT / "nfl_backtest_resumen.csv"
CALIBRATION = ROOT / "nfl_calibracion.json"
MIN_CALIBRATION_N = 100
SUMMARY_COLUMNS = [
    "mercado", "n_train", "n_test", "brier_sin_calibrar", "log_loss_sin_calibrar",
    "brier_calibrado", "log_loss_calibrado", "brier_baseline_50", "log_loss_baseline_50",
    "tasa_real", "probabilidad_media", "sesgo", "calibracion", "precio_kalshi_comparable", "nota",
]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _logloss(probabilities, outcomes):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def _brier(probabilities, outcomes):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def fit_platt(probabilities, outcomes):
    x = _logit(probabilities)
    y = np.asarray(outcomes, dtype=float)

    def objective(params):
        intercept, slope = params
        z = np.clip(intercept + slope * x, -35, 35)
        # L2 regularization prevents extreme parameters on quasi-separated samples.
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.001 * (intercept ** 2 + slope ** 2))

    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError("No se pudo ajustar el calibrador Platt.")
    intercept, slope = [float(x) for x in result.x]
    if not math.isfinite(intercept) or not math.isfinite(slope):
        raise RuntimeError("El calibrador produjo parámetros no finitos.")
    return intercept, slope


def apply_platt(probabilities, intercept, slope):
    z = np.clip(intercept + slope * _logit(probabilities), -35, 35)
    return 1 / (1 + np.exp(-z))


def build_walk_forward(max_games=500):
    all_games = nfl._nflverse_games().sort_values("gameday").reset_index(drop=True)
    games = all_games.tail(max_games).reset_index(drop=True) if max_games > 0 else all_games
    observations = {"winner": [], "spread_-3.5": [], "total_over_44.5": [], "team_total_home_over_21.5": []}
    for index, row in games.iterrows():
        kickoff = pd.Timestamp(row["gameday"])
        prior = all_games[all_games["gameday"] < kickoff]
        if len(prior) < 100:
            continue
        home = nfl._canonical_team_code(row["home_team"])
        away = nfl._canonical_team_code(row["away_team"])
        if home not in nfl.TEAM_ALIASES or away not in nfl.TEAM_ALIASES:
            continue
        model = nfl.build_team_model(all_games, as_of=kickoff.to_pydatetime())
        if model["teams"][home]["sample"] < 4 or model["teams"][away]["sample"] < 4:
            continue
        forecast = nfl.predict_game(model, home, away, kickoff=kickoff.to_pydatetime())
        if not forecast:
            continue
        home_score, away_score = float(row["home_score"]), float(row["away_score"])
        margin, total = home_score - away_score, home_score + away_score
        if margin != 0:
            observations["winner"].append({
                "date": kickoff.isoformat(), "p": forecast["home_win"], "y": 1.0 if margin > 0 else 0.0,
            })
        margin_values, margin_probs = forecast["margin_values"], forecast["margin_probs"]
        observations["spread_-3.5"].append({
            "date": kickoff.isoformat(), "p": float(margin_probs[margin_values > 3.5].sum()),
            "y": 1.0 if margin > 3.5 else 0.0,
        })
        total_values, total_probs = forecast["total_values"], forecast["total_probs"]
        observations["total_over_44.5"].append({
            "date": kickoff.isoformat(), "p": float(total_probs[total_values > 44.5].sum()),
            "y": 1.0 if total > 44.5 else 0.0,
        })
        home_points_values, home_points_probs = nfl._normal_pmf(forecast["home_points_mean"], 9.5, 0, 70)
        observations["team_total_home_over_21.5"].append({
            "date": kickoff.isoformat(), "p": float(home_points_probs[home_points_values > 21.5].sum()),
            "y": 1.0 if home_score > 21.5 else 0.0,
        })
    return observations


def summarize(observations):
    summary, calibrators = [], {}
    for market, items in observations.items():
        items = sorted(items, key=lambda item: item["date"])
        n = len(items)
        if n < 2:
            summary.append({
                "mercado": market, "n_train": 0, "n_test": n,
                "brier_sin_calibrar": "", "log_loss_sin_calibrar": "",
                "brier_calibrado": "", "log_loss_calibrado": "",
                "brier_baseline_50": "", "log_loss_baseline_50": "",
                "tasa_real": "", "probabilidad_media": "", "sesgo": "",
                "calibracion": "calibración insuficiente",
                "precio_kalshi_comparable": "No disponible: sin snapshot prepartido",
                "nota": "Muestra histórica insuficiente.",
            })
            continue
        split = max(1, int(n * 0.70))
        train, test = items[:split], items[split:]
        if not test:
            train, test = items[:-1], items[-1:]
        p_train = np.array([item["p"] for item in train], dtype=float)
        y_train = np.array([item["y"] for item in train], dtype=float)
        p_test = np.array([item["p"] for item in test], dtype=float)
        y_test = np.array([item["y"] for item in test], dtype=float)
        p_cal_test = None
        status = "calibración insuficiente"
        if len(train) >= MIN_CALIBRATION_N and len(np.unique(y_train)) == 2:
            try:
                a_test, b_test = fit_platt(p_train, y_train)
                p_cal_test = apply_platt(p_test, a_test, b_test)
                # Production coefficients use all walk-forward observations after evaluation is calculated.
                a_full, b_full = fit_platt(np.array([item["p"] for item in items]), np.array([item["y"] for item in items]))
                if market in {"winner", "spread_-3.5", "total_over_44.5"}:
                    calibrators[market] = {
                        "intercept": a_full, "slope": b_full, "n_train": n,
                        "version": f"platt_{market}_n{n}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                    }
                status = f"Platt entrenado n={n}; evaluación temporal separada"
            except (RuntimeError, ValueError, FloatingPointError):
                status = "calibración insuficiente: ajuste no estable"
        p_mean = float(np.mean(p_test))
        y_mean = float(np.mean(y_test))
        if p_mean > y_mean + 0.02:
            bias = "sobreestima"
        elif p_mean < y_mean - 0.02:
            bias = "subestima"
        else:
            bias = "sin sesgo claro en esta muestra"
        summary.append({
            "mercado": market, "n_train": len(train), "n_test": len(test),
            "brier_sin_calibrar": round(_brier(p_test, y_test), 6),
            "log_loss_sin_calibrar": round(_logloss(p_test, y_test), 6),
            "brier_calibrado": round(_brier(p_cal_test, y_test), 6) if p_cal_test is not None else "",
            "log_loss_calibrado": round(_logloss(p_cal_test, y_test), 6) if p_cal_test is not None else "",
            "brier_baseline_50": round(_brier(np.full(len(y_test), 0.5), y_test), 6),
            "log_loss_baseline_50": round(_logloss(np.full(len(y_test), 0.5), y_test), 6),
            "tasa_real": round(y_mean, 6), "probabilidad_media": round(p_mean, 6), "sesgo": bias,
            "calibracion": status,
            "precio_kalshi_comparable": "No disponible: el endpoint de mercados liquidados no conserva un snapshot prepartido fiable",
            "nota": "Backtest walk-forward. Spread usa línea proxy local -3.5; total usa Over 44.5; total por equipo usa local Over 21.5. No son contratos Kalshi históricos.",
        })
    return summary, calibrators


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.force and CALIBRATION.exists():
        try:
            current = json.loads(CALIBRATION.read_text(encoding="utf-8"))
            generated = datetime.fromisoformat(current.get("generated_at", "").replace("Z", "+00:00"))
            if generated.date() == datetime.now(timezone.utc).date():
                print(f"BACKTEST_NFL_CACHE: calibración ya generada hoy {current.get('generated_at')}; no se recalcula.")
                return
        except (ValueError, OSError, TypeError):
            pass
    observations = build_walk_forward(max_games=args.max_games)
    summary, calibrators = summarize(observations)
    payload = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "nflverse games.csv (resultados históricos)",
        "minimum_sample": MIN_CALIBRATION_N,
        "markets": calibrators,
        "limitation": "No se encontraron snapshots prepartido de precios Kalshi en el endpoint público usado; spread/total son líneas proxy.",
    }
    tmp = CALIBRATION.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(CALIBRATION)
    nfl._write_report(SUMMARY, SUMMARY_COLUMNS, summary)
    print("BACKTEST_NFL_RESUMEN")
    print(pd.DataFrame(summary, columns=SUMMARY_COLUMNS).to_csv(index=False))
    print(f"BACKTEST_NFL: juegos_analizados={max((len(v) for v in observations.values()), default=0)}; calibradores={list(calibrators)}; archivo={SUMMARY.name}; mínimo_muestra={MIN_CALIBRATION_N}")


if __name__ == "__main__":
    main()
