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
    assert {row["bloque"] for row in rows}.issubset({"SENCILLAS", "ALTA_PROBABILIDAD", "A_TENER_EN_CUENTA"})
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
