import unittest
import numpy as np
from scipy.stats import poisson

from laliga_engine import modelar_seleccion_kalshi as modelar_laliga
from champions_engine import modelar_seleccion_kalshi as modelar_champions


class FirstGoalModelGateTests(unittest.TestCase):
    def setUp(self):
        self.axis = np.arange(0, 9)
        self.i, self.j = np.meshgrid(self.axis, self.axis, indexing="ij")
        self.p_goals = np.outer(poisson.pmf(self.axis, 1.4), poisson.pmf(self.axis, 1.1))
        self.p_goals /= self.p_goals.sum()

    def test_laliga_does_not_infer_first_scorer_from_full_time_score(self):
        for side in ("local", "visita"):
            with self.subTest(side=side):
                result = modelar_laliga(
                    ("KXLALIGAFTTS", side, None, None),
                    "Local FC", "Visitante FC", self.p_goals, self.i, self.j,
                    p_corners=None, ci=None, cj=None,
                )
                self.assertIsNone(result)

    def test_champions_does_not_infer_first_scorer_from_full_time_score(self):
        for side in ("local", "visita"):
            with self.subTest(side=side):
                result = modelar_champions(
                    ("KXUCLFTTS", side, None, None),
                    "Local FC", "Visitante FC", self.p_goals, self.i, self.j,
                    p_corners=None, ci=None, cj=None,
                )
                self.assertIsNone(result)

    def test_no_goal_probability_remains_modelable(self):
        result = modelar_laliga(
            ("KXLALIGAFTTS", "sin_goles", None, None),
            "Local FC", "Visitante FC", self.p_goals, self.i, self.j,
            p_corners=None, ci=None, cj=None,
        )
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result[2], 0)
        self.assertLessEqual(result[2], 1)


if __name__ == "__main__":
    unittest.main()
