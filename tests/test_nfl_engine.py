import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import nfl_engine as nfl


def sample_games():
    return pd.DataFrame([
        {"season": 2024, "week": 1, "gameday": "2024-09-08T17:00:00Z", "home_team": "KC", "away_team": "BAL", "home_score": 27, "away_score": 20},
        {"season": 2024, "week": 2, "gameday": "2024-09-15T17:00:00Z", "home_team": "BAL", "away_team": "LV", "home_score": 23, "away_score": 16},
        {"season": 2024, "week": 3, "gameday": "2024-09-22T17:00:00Z", "home_team": "KC", "away_team": "LV", "home_score": 28, "away_score": 14},
    ])


def test_model_probabilities_are_finite_and_bounded():
    model = nfl.build_team_model(sample_games(), as_of="2025-01-01T00:00:00Z")
    pred = nfl.predict_game(model, "KC", "BAL")
    assert pred is not None
    for key in ("home_win", "away_win"):
        assert math.isfinite(pred[key])
        assert 0 <= pred[key] <= 1
    assert abs(pred["home_win"] + pred["away_win"] - 1) < 1e-9
    assert np.isfinite(pred["margin_probs"]).all()
    assert abs(float(pred["margin_probs"].sum()) - 1) < 1e-9
    assert np.isfinite(pred["total_probs"]).all()
    assert abs(float(pred["total_probs"].sum()) - 1) < 1e-9


def test_market_semantics_are_conservative():
    game = {"home": "KC", "away": "BAL"}
    pred = nfl.predict_game(nfl.build_team_model(sample_games(), as_of="2025-01-01T00:00:00Z"), "KC", "BAL")
    winner = {"ticker": "KXNFLGAME-EXAMPLE-KC", "title": "Will Kansas City Chiefs win?", "yes_sub_title": "Kansas City Chiefs"}
    p, reason = nfl.model_market(winner, "KXNFLGAME", game, pred)
    assert p is not None and 0 <= p <= 1
    unknown = {"ticker": "KXNFLPLAYER-EXAMPLE", "title": "Will Patrick Mahomes pass for 300+ yards?"}
    p, reason = nfl.model_market(unknown, "KXNFLPLAYERPASSYDS", game, pred)
    assert p is None
    assert "prop de jugador" in reason


def test_nfl_integer_margin_key_numbers_do_not_break_normalization():
    values, probs = nfl._normal_pmf(2.7, 13.5, -45, 45, key_numbers=(-14, -10, -7, -6, -3, 3, 6, 7, 10, 14))
    assert len(values) == len(probs)
    assert np.all(probs >= 0)
    assert np.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-9


def test_all_probability_values_in_saved_candidates_are_valid():
    model = nfl.build_team_model(sample_games(), as_of="2025-01-01T00:00:00Z")
    pred = nfl.predict_game(model, "KC", "BAL")
    for key in ("home_win", "away_win"):
        assert nfl.valid_probability(pred[key]) is not None



def test_kalshi_event_date_uses_et_date_and_espn_fixture_for_exact_kickoff():
    event = {
        "event_ticker": "KXNFLGAME-26OCT11CINMIA",
        "title": "CIN Bengals vs MIA Dolphins",
        "sub_title": "CIN vs MIA (Oct 11)",
    }
    assert nfl._kickoff(event) is None
    assert str(nfl._event_date(event)) == "2026-10-11"
    fixtures = [{
        "home": "MIA", "away": "CIN", "date": "2026-10-11T17:00:00Z",
        "match_id": "NFL:2026-10-11:CIN@MIA",
    }]
    matched = nfl._match_catalog_event(event, nfl._event_teams(event), fixtures, None)
    assert matched is fixtures[0]



def test_kalshi_total_threshold_semantics_are_exact():
    inclusive = {"ticker": "KXNFLTOTAL-EXAMPLE", "yes_sub_title": "Over 25+ points"}
    exclusive = {"ticker": "KXNFLTOTAL-EXAMPLE", "yes_sub_title": "Over 25.5 points"}
    assert nfl._line_from_market(inclusive) == 25
    assert nfl._yes_outcome(inclusive, "total") == "over_ge"
    assert nfl._line_from_market(exclusive) == 25.5
    assert nfl._yes_outcome(exclusive, "total") == "over_gt"


def test_generate_candidates_assigns_independent_blocks_and_keeps_side():
    model = nfl.build_team_model(sample_games(), as_of="2025-01-01T00:00:00Z")
    event = {
        "event_ticker": "KXNFLGAME-26OCT18CINMIA",
        "series_ticker": "KXNFLGAME",
        "title": "CIN Bengals vs MIA Dolphins",
        "sub_title": "CIN vs MIA (Oct 18)",
        "markets": [
            {"ticker": "KXNFLGAME-26OCT18CINMIA-MIA", "series_ticker": "KXNFLGAME",
             "title": "Miami wins", "yes_ask_dollars": "0.30", "no_ask_dollars": "0.30"},
            {"ticker": "KXNFLGAME-26OCT18CINMIA-CIN", "series_ticker": "KXNFLGAME",
             "title": "Cincinnati wins", "yes_ask_dollars": "0.30", "no_ask_dollars": "0.30"},
            {"ticker": "KXNFLTOTAL-26OCT18CINMIA-OVER30.5", "series_ticker": "KXNFLTOTAL",
             "title": "Over 30.5 points", "yes_ask_dollars": "0.30", "no_ask_dollars": "0.30"},
        ],
    }
    fixtures = [{
        "home": "MIA", "away": "CIN", "date": "2026-10-18T17:00:00Z",
        "match_id": "NFL:2026-10-18:CIN@MIA",
    }]
    rows, rejected, catalog, coverage = nfl.generate_candidates(
        [event], model, now="2026-10-10T12:00:00Z", upcoming_games=fixtures
    )
    assert catalog
    assert coverage and coverage[0]["match_id"] == "NFL:2026-10-18:CIN@MIA"
    assert rows and {row["lado"] for row in rows}.issubset({"YES", "NO"})
    assert {row["bloque"] for row in rows} == {"SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA"}
    total_yes = [row for row in rows if row["ticker"] == "KXNFLTOTAL-26OCT18CINMIA-OVER30.5" and row["lado"] == "YES"]
    assert {row["bloque"] for row in total_yes} == {"SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA"}
    assert len({row["prediction_id"] for row in total_yes}) == 3
    ids = [row["prediction_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert all(row["descripcion"].startswith(("SÍ:", "NO:")) for row in rows)


def test_atomic_save_preserves_yes_no_side_and_freezes_duplicates(tmp_path):
    future = "2030-10-20T17:00:00+00:00"
    captured = "2026-10-10T12:00:00+00:00"
    common = {
        "schema_version": "1", "event_ticker": "KXNFLGAME-30OCT20CINMIA",
        "ticker": "KXNFLGAME-30OCT20CINMIA-MIA", "match_id": "NFL:2030-10-20:CIN@MIA",
        "fecha": future, "hora_inicio": future, "local": "MIA", "visita": "CIN",
        "categoria": "winner", "mercado": "Miami wins", "line": "", "equipo_yes": "MIA",
        "tipo_salida": "winner", "descripcion": "SÍ: los Miami Dolphins ganan el partido",
        "bloque": "SENCILLAS", "probabilidad_original": "0.6", "probabilidad_calibrada": "",
        "version_calibracion": "calibración insuficiente", "probabilidad_modelo": "0.6",
        "precio_captura": "0.3", "probabilidad_implicita": "0.3", "edge": "0.3",
        "revision": "REVISAR", "estado_modelo": "test", "reglas": "test",
        "capturado_en": captured, "disponibilidad": "Disponible en Kalshi",
    }
    yes = {**common, "prediction_id": "pred-yes", "lado": "YES", "seleccion": "YES"}
    no = {**common, "prediction_id": "pred-no", "lado": "NO", "seleccion": "NO",
          "descripcion": "NO: los Miami Dolphins no ganan el partido"}
    path = tmp_path / "predicciones_nfl.csv"
    first = nfl.save_capture([yes, no], path=path)
    second = nfl.save_capture([yes, no], path=path)
    assert first["accepted"] == 2
    assert second["accepted"] == 0
    saved = pd.read_csv(path, dtype=str).fillna("")
    assert set(saved["lado"]) == {"YES", "NO"}
    assert len(saved) == 2



def test_espn_fallback_settles_finished_winner_by_teams_and_date(monkeypatch):
    nfl.ESPN_SCOREBOARD_CACHE.clear()
    event = {
        "id": "espn-game-1",
        "date": "2026-10-11T17:00:00Z",
        "status": {"type": {"name": "STATUS_FINAL", "completed": True}},
        "competitions": [{
            "date": "2026-10-11T17:00:00Z",
            "competitors": [
                {"homeAway": "home", "team": {"abbreviation": "MIA"}, "score": "24"},
                {"homeAway": "away", "team": {"abbreviation": "CIN"}, "score": "17"},
            ],
        }],
    }
    def fake_get(url, params=None):
        return {"events": [event] if params and params.get("dates") == "20261011" else []}
    monkeypatch.setattr(nfl, "_get", fake_get)
    row = {
        "hora_inicio": "2026-10-11T17:00:00Z", "fecha": "2026-10-11T17:00:00Z",
        "local": "MIA", "visita": "CIN", "categoria": "winner", "equipo_yes": "MIA",
        "tipo_salida": "winner", "lado": "YES", "ticker": "KXNFLGAME-26OCT11CINMIA-MIA",
    }
    settled = nfl.settle_from_espn(row)
    assert settled and settled["state"] == "WIN"
    row["lado"] = "NO"
    settled_no = nfl.settle_from_espn(row)
    assert settled_no and settled_no["state"] == "LOSS"
    assert "24" in settled["score"] and "17" in settled["score"]



def test_calibration_requires_minimum_sample_and_preserves_probability_bounds():
    p, version, status = nfl.calibrate_probability(0.73, "winner", None, "winner", calibration={})
    assert p is None and version is None and status == "calibración insuficiente"
    calibration = {"markets": {"winner": {
        "n_train": 150, "intercept": 0.1, "slope": 0.9, "version": "platt_winner_n150",
    }}}
    calibrated, version, status = nfl.calibrate_probability(0.73, "winner", None, "winner", calibration=calibration)
    assert 0 <= calibrated <= 1
    assert version == "platt_winner_n150"
    assert status == "aplicada"



def test_csv_reader_quarantines_bad_rows_and_migrates_missing_schema(tmp_path):
    path = tmp_path / "predicciones_nfl.csv"
    path.write_text("prediction_id,ticker\n1,KXNFLGAME-TEST\n2,KXNFLGAME-TEST,EXTRA\n", encoding="utf-8")
    rows = nfl.read_csv_tolerant(path, expected_columns=["prediction_id", "ticker", "schema_version"])
    assert len(rows) == 1
    assert rows[0]["schema_version"] == "0"
    quarantine = tmp_path / "predicciones_nfl_filas_malas.csv"
    assert quarantine.exists()
    bad = pd.read_csv(quarantine, dtype=str)
    assert "columnas" in bad.loc[0, "motivo"]
