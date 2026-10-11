import unittest
import numpy as np
from scipy.stats import poisson

from laliga_engine import modelar_seleccion_kalshi


class LaLigaSourceGateTests(unittest.TestCase):
    def test_corners_market_is_not_modelled_without_corner_data(self):
        axis = np.arange(0, 9)
        i, j = np.meshgrid(axis, axis, indexing="ij")
        p_goals = np.outer(poisson.pmf(axis, 1.4), poisson.pmf(axis, 1.1))
        p_goals /= p_goals.sum()
        result = modelar_seleccion_kalshi(
            ("KXLALIGACORNERS", "over", 3.0, "over"),
            "Local FC", "Visitante FC",
            p_goals, i, j,
            p_corners=None, ci=None, cj=None,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
