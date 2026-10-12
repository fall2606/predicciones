import unittest
import numpy as np
import pandas as pd
from scipy.stats import poisson

from laliga_engine import modelar_seleccion_kalshi, calibrar_resumen


class LaLigaSourceGateTests(unittest.TestCase):
    def test_category_calibration_corrects_overconfidence_before_market_anchor(self):
        predictions = pd.DataFrame([
            {
                "prediction_id": f"p-{i}",
                "categoria": "Goles totales",
                "probabilidad_base": 0.80,
                "capturado_en": f"2026-10-{i + 1:02d}T10:00:00+00:00",
                "ticker": f"KX-TOTAL-{i}",
            }
            for i in range(15)
        ])
        results = pd.DataFrame([
            {"prediction_id": f"p-{i}", "estado": "LOSS" if i < 12 else "WIN",
             "resuelto_en": f"2026-10-{i + 1:02d}T12:00:00+00:00"}
            for i in range(15)
        ])
        calibrated = calibrar_resumen(
            [{"Tipo": "Goles totales", "Probabilidad base": 0.80,
              "Probabilidad del modelo": "80%", "Precio de compra": "60%"}],
            predictions,
            results,
        )[0]
        self.assertLess(float(calibrated["probabilidad_calibrada"]), 0.80)
        self.assertAlmostEqual(float(calibrated["probabilidad_calibrada"]), (3 + 20 * 0.80) / 35)
        self.assertEqual(calibrated["Calibración"], "Aprendida con 15 resultados")

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
