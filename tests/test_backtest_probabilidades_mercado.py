import unittest
from scripts.backtest_probabilidades_mercado import build_backtest


def prediction(pid, category, model, price, kickoff="2026-10-20T00:00:00+00:00",
               captured="2026-10-19T00:00:00+00:00"):
    return {
        "prediction_id": pid, "categoria": category, "probabilidad_base": str(model),
        "probabilidad_calibrada": str(model), "precio": str(price),
        "ticker": f"KX-REAL-{pid}", "fecha": kickoff, "capturado_en": captured,
    }


def result(pid, state="WIN"):
    return {"prediction_id": pid, "estado": state, "resuelto_en": "2026-10-21T00:00:00+00:00"}


class ChronologicalMarketBacktestTests(unittest.TestCase):
    def test_only_pre_kickoff_paired_rows_and_validated_weight_when_brier_improves(self):
        predictions = [
            prediction(f"good-{i}", "SENCILLAS · Hándicap de goles", 0.80, 0.50)
            for i in range(30)
        ] + [
            prediction(f"bad-{i}", "SENCILLAS · Resultado", 0.20, 0.60)
            for i in range(30)
        ]
        predictions.append(prediction(
            "late", "SENCILLAS · Hándicap de goles", 0.99, 0.20,
            captured="2026-10-20T01:00:00+00:00",
        ))
        results = [result(f"good-{i}") for i in range(30)]
        results += [result(f"bad-{i}") for i in range(30)]
        results.append(result("late"))

        report = build_backtest(predictions, results)

        self.assertEqual(report["paired_rows"], 60)
        self.assertEqual(report["excluded_rows"]["not_pre_kickoff"], 1)
        good = report["categories"]["hándicap de goles"]
        bad = report["categories"]["resultado"]
        self.assertEqual(good["n"], 30)
        self.assertEqual(good["peso_modelo_recomendado"], 0.25)
        self.assertLess(good["brier_blend_25pct"], good["brier_price"])
        self.assertEqual(bad["peso_modelo_recomendado"], 0.05)
        self.assertGreater(bad["brier_blend_25pct"], bad["brier_price"])

    def test_missing_price_never_enters_paired_brier(self):
        p = prediction("no-price", "Goles totales", 0.8, "—")
        report = build_backtest([p], [result("no-price")])
        self.assertEqual(report["paired_rows"], 0)
        self.assertEqual(report["excluded_rows"]["invalid_price"], 1)


if __name__ == "__main__":
    unittest.main()
