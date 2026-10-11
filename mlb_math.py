"""Numerically safe MLB model helpers shared by the Streamlit page and tests."""
import math
import numpy as np
from scipy.stats import nbinom, poisson


def pitcher_stats_from_api(stat, name="Pitcher probable"):
    """Return pitcher inputs only when all API numbers are finite and physically valid."""
    try:
        values = {
            "era": float(stat.get("era")),
            "innings": float(stat.get("inningsPitched") or 0),
            "strikeouts": float(stat.get("strikeOuts") or 0),
            "hits_allowed": float(stat.get("hits") or 0),
            "walks": float(stat.get("baseOnBalls") or 0),
            "earned_runs": float(stat.get("earnedRuns") or 0),
        }
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in values.values()):
        return None
    if values["era"] <= 0 or any(v < 0 for k, v in values.items() if k != "era"):
        return None
    return {"nombre": name or "Pitcher probable", **values}


def score_probability_matrix(axis, mu_home, mu_away):
    """Build a finite, normalized score matrix; reject invalid pitcher-derived means."""
    axis = np.asarray(axis, dtype=float)
    def valid_mean(value):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return 4.4
        return number if math.isfinite(number) and 0.05 <= number <= 20 else 4.4

    dispersion = 0.12
    n = 1 / dispersion
    def pmf(mean):
        p = n / (n + mean)
        return nbinom.pmf(axis, n, p)

    matrix = np.outer(pmf(valid_mean(mu_home)), pmf(valid_mean(mu_away)))
    total = float(np.sum(matrix))
    if not np.all(np.isfinite(matrix)) or not math.isfinite(total) or total <= 0:
        matrix = np.outer(poisson.pmf(axis, 4.4), poisson.pmf(axis, 4.4))
        total = float(np.sum(matrix))
    if not np.all(np.isfinite(matrix)) or not math.isfinite(total) or total <= 0:
        raise ValueError("No se pudo construir una matriz de probabilidad MLB finita.")
    matrix = matrix / total
    if not np.all(np.isfinite(matrix)):
        raise ValueError("La matriz de probabilidad MLB contiene valores no finitos.")
    return matrix
