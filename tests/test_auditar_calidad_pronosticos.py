import unittest

from scripts.auditar_calidad_pronosticos import parse_probability, score_rows, summarize_active_pending, summarize_reconciliation, reconciliation_coverage


class PredictionQualityAuditTests(unittest.TestCase):
    def test_probability_parser_rejects_invalid_and_accepts_explicit_percent(self):
        self.assertEqual(parse_probability("72%"), 0.72)
        self.assertEqual(parse_probability("0.72"), 0.72)
        self.assertIsNone(parse_probability("72"))
        self.assertIsNone(parse_probability("NaN"))
        self.assertIsNone(parse_probability("-0.1"))
        self.assertIsNone(parse_probability("1.2"))

    def test_scores_only_terminal_win_loss_and_joins_by_prediction_id(self):
        predictions = [
            {"prediction_id": "a", "probabilidad_calibrada": "0.8"},
            {"prediction_id": "b", "probabilidad_calibrada": "0.3"},
            {"prediction_id": "c", "probabilidad_calibrada": "0.9"},
            {"prediction_id": "d", "probabilidad_calibrada": "bad"},
        ]
        results = [
            {"prediction_id": "a", "estado": "WIN", "resuelto_en": "2026-10-01"},
            {"prediction_id": "b", "estado": "LOSS", "resuelto_en": "2026-10-01"},
            {"prediction_id": "c", "estado": "ANULADA", "resuelto_en": "2026-10-01"},
        ]
        report = score_rows(predictions, results, ("probabilidad_calibrada",))
        self.assertEqual(report["n_evaluable"], 2)
        self.assertEqual(report["capturas_con_resultado_terminal"], 3)
        self.assertEqual(report["sin_resultado_terminal"], 1)
        self.assertAlmostEqual(report["brier"], ((0.8-1)**2 + (0.3-0)**2)/2)
        self.assertIsNotNone(report["log_loss"])
        self.assertEqual(sum(b["n"] for b in report["calibracion_intervalos"]), 2)

    def test_latest_result_for_same_id_wins_and_duplicates_are_reported(self):
        predictions = [
            {"prediction_id": "a", "probabilidad_modelo": "0.7"},
            {"prediction_id": "a", "probabilidad_modelo": "0.7"},
        ]
        results = [
            {"prediction_id": "a", "estado": "WIN", "resuelto_en": "2026-10-01"},
            {"prediction_id": "a", "estado": "LOSS", "resuelto_en": "2026-10-02"},
        ]
        report = score_rows(predictions, results, ("probabilidad_modelo",))
        self.assertEqual(report["n_evaluable"], 2)
        self.assertEqual(report["prediction_id_duplicados_en_capturas"], 1)
        self.assertEqual(report["ids_resultado_duplicados"], 1)
        self.assertEqual(report["tasa_acierto"], 0.0)

    def test_reconciliation_audit_preserves_alert_reasons(self):
        rows = [
            {"deporte": "LaLiga", "clasificacion": "PENDIENTE_ACTIVA", "cuenta_alerta": "SI", "motivo": "sin ticker"},
            {"deporte": "LaLiga", "clasificacion": "PENDIENTE_ACTIVA", "cuenta_alerta": "SI", "motivo": "sin ticker"},
            {"deporte": "MLB", "clasificacion": "LIQUIDADA", "cuenta_alerta": "NO", "motivo": "resultado registrado: WIN"},
        ]
        report = summarize_reconciliation(rows)
        alerts = [row for row in report if row["cuenta_alerta"] == "SI"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["filas"], 2)
        self.assertEqual(alerts[0]["motivo"], "sin ticker")

    def test_active_pending_uses_canonical_classification_not_all_missing_results(self):
        rows = [
            {"deporte": "LaLiga", "clasificacion": "PENDIENTE_ACTIVA", "cuenta_alerta": "SI"},
            {"deporte": "LaLiga", "clasificacion": "NO_FINALIZADA", "cuenta_alerta": "NO"},
            {"deporte": "MLB", "clasificacion": "HISTORICO_NO_PARTIDO", "cuenta_alerta": "NO"},
            {"deporte": "MLB", "clasificacion": "PENDIENTE_ACTIVA", "cuenta_alerta": "NO"},
            {"deporte": "Champions", "clasificacion": "PENDIENTE_ACTIVA", "cuenta_alerta": "SI"},
        ]
        self.assertEqual(summarize_active_pending(rows), {"LaLiga": 1, "Champions": 1})

    def test_missing_results_produce_no_fake_score(self):
        report = score_rows([{"prediction_id": "pending", "probabilidad_modelo": "0.6"}], [], ("probabilidad_modelo",))
        self.assertEqual(report["n_evaluable"], 0)
        self.assertIsNone(report["brier"])
        self.assertEqual(report["sin_resultado_terminal"], 1)



    def test_pending_count_is_unavailable_when_reconciliation_coverage_is_incomplete(self):
        predictions = [{"prediction_id": "a"}, {"prediction_id": "b"}]
        reconciliation = [
            {"deporte": "Champions", "prediction_id": "a",
             "clasificacion": "LIQUIDADA", "cuenta_alerta": "NO"}
        ]
        coverage = reconciliation_coverage(predictions, reconciliation, "Champions")
        self.assertFalse(coverage["reconciliacion_cobertura_completa"])
        self.assertEqual(coverage["prediction_ids_sin_fila_reconciliacion"], 1)
        self.assertEqual(coverage["filas_reconciliacion"], 1)

    def test_pending_count_is_verifiable_only_when_every_prediction_has_a_reconciliation_row(self):
        predictions = [{"prediction_id": "a"}, {"prediction_id": "b"}]
        reconciliation = [
            {"deporte": "Champions", "prediction_id": "a"},
            {"deporte": "Champions", "prediction_id": "b"},
        ]
        coverage = reconciliation_coverage(predictions, reconciliation, "Champions")
        self.assertTrue(coverage["reconciliacion_cobertura_completa"])
        self.assertEqual(coverage["prediction_ids_sin_fila_reconciliacion"], 0)


if __name__ == "__main__":
    unittest.main()
