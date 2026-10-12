import csv, tempfile, unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
import prediction_core as core

class CoreTests(unittest.TestCase):
 def test_probability_validation(self):
  for v in ("NaN","inf","-inf",-0.1,1.1,None,""): self.assertIsNone(core.valid_probability(v))
  self.assertEqual(core.valid_probability(0),0); self.assertEqual(core.valid_probability(1),1)

 def test_price_parser_and_edge_are_conservative(self):
  self.assertEqual(core.probability_fraction("57%"), 0.57)
  self.assertEqual(core.probability_fraction("$0.57"), 0.57)
  self.assertEqual(core.probability_fraction(0.57), 0.57)
  self.assertIsNone(core.probability_fraction("—"))
  self.assertIsNone(core.probability_fraction("NaN"))
  self.assertIsNone(core.probability_fraction(0))
  self.assertAlmostEqual(core.calculate_edge("72%", "57%"), 0.15)
  self.assertIsNone(core.calculate_edge("72%", "—"))

 def test_market_anchor_uses_market_as_base_and_caps_model_weight(self):
  self.assertAlmostEqual(core.anchor_probability_to_market(0.80, 0.60), 0.65)
  self.assertAlmostEqual(core.anchor_probability_to_market(0.20, 0.60), 0.50)
  self.assertAlmostEqual(core.anchor_probability_to_market(0.80, 0.60, weight=0.90), 0.65)
  self.assertIsNone(core.anchor_probability_to_market(0.80, "—"))
  self.assertIsNone(core.anchor_probability_to_market("—", 0.60))

 def test_rank_single_candidates_preserves_yes_no_and_uses_net_calibrated_edge(self):
  row={"Ticker":"KXLALIGAGAME-TEST-AWAY","Apuesta sencilla":"Gana Celta","Categoría resumen":"Resultado",
       "Probabilidad modelo":"70%","probabilidad_calibrada":"0.70","Precio de compra":"40%","Ventaja (edge)":-0.30,"_lado":"YES"}
  selected=core.rank_single_candidates([row],limit=1)
  self.assertEqual(len(selected),1)
  self.assertEqual(selected[0].get("_lado"),"YES")
  self.assertAlmostEqual(selected[0]["Ventaja (edge)"],0.065)
  self.assertAlmostEqual(selected[0]["probabilidad_final"],0.475)
  self.assertEqual(selected[0]["Probabilidad del modelo"],"70.0%")
  self.assertEqual(selected[0]["Probabilidad final"],"47.5%")
  calibrated={**row,"probabilidad_calibrada":"0.60","Precio de compra":"65%"}
  self.assertEqual(core.rank_single_candidates([calibrated]),[])

 def test_legacy_missing_side_is_inferred_only_for_unambiguous_game_and_btts(self):
  game={"ticker":"KXLALIGAGAME-TEST-AWAY","local":"Elche","visita":"Celta","jugada":"Gana Celta",
        "seleccion":"","clave_interna":"(('KXLALIGAGAME', 'visita', None, None), None)"}
  btts={"ticker":"KXLALIGABTTS-TEST","local":"Elche","visita":"Celta",
        "jugada":"Uno o ambos equipos se quedan sin marcar","seleccion":"",
        "clave_interna":"(('KXLALIGABTTS', 'ambos', None, None), None)"}
  spread={"ticker":"KXLALIGASPREAD-TEST","local":"Elche","visita":"Celta",
          "jugada":"Elche no pierde por más de 1.5 goles","seleccion":"",
          "clave_interna":"(('KXLALIGASPREAD', 'visita', 1.5, None), None)"}
  self.assertEqual(core._side_from_row(game,"lado"),"YES")
  self.assertEqual(core._side_from_row(btts,"lado"),"NO")
  self.assertIsNone(core._side_from_row(spread,"lado"))

 def test_legacy_team_corners_side_is_inferred_only_from_exact_range_and_strike(self):
  row={"ticker":"KXLALIGATCORNERS-26OCT11RBBOSA-RBB6","local":"Real Betis","visita":"Osasuna",
       "jugada":"0–5 corners de Real Betis","line":"6+","seleccion":"",
       "clave_interna":"(('KXLALIGATCORNERS', 'local', 6.0, 'over'), None)"}
  self.assertEqual(core._side_from_row(row,"lado"),"NO")
  ambiguous={**row,"jugada":"Corners de Real Betis","clave_interna":"(('KXLALIGATCORNERS', 'local', 6.0, 'over'), None)"}
  self.assertIsNone(core._side_from_row(ambiguous,"lado"))
 def test_freeze_and_probability_gate(self):
  now=datetime.now(timezone.utc)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"p.csv"; cols=["prediction_id","ticker","fecha","capturado_en","p"]
   future=(now+timedelta(hours=2)).isoformat()
   rows=[{"prediction_id":"ok","ticker":"KXREAL-1","fecha":future,"capturado_en":now.isoformat(),"p":"0.72"},
    {"prediction_id":"late","ticker":"KXREAL-2","fecha":(now-timedelta(minutes=1)).isoformat(),"capturado_en":(now-timedelta(minutes=2)).isoformat(),"p":"0.6"},
    {"prediction_id":"fake","ticker":"—","fecha":future,"capturado_en":now.isoformat(),"p":"0.6"},
    {"prediction_id":"nan","ticker":"KXREAL-3","fecha":future,"capturado_en":now.isoformat(),"p":"NaN"}]
   self.assertEqual(core.append_immutable_predictions(p,cols,rows,probability_field="p"),{"accepted":1,"rejected":3})
   with p.open(encoding="utf-8") as f: saved=list(csv.DictReader(f))
   self.assertEqual(saved[0]["prediction_id"],"ok"); self.assertEqual(saved[0]["p"],"0.72000000")
   self.assertTrue(saved[0]["capturado_en"])
   self.assertEqual(saved[0]["fecha"], future)
   self.assertEqual(core.append_immutable_predictions(p,cols,[rows[0]],probability_field="p")["accepted"],0)
 def test_recovery_old_yes_no_void_and_idempotence(self):
  with tempfile.TemporaryDirectory() as d:
   p,r=Path(d)/"p.csv",Path(d)/"r.csv"; old=(datetime.now(timezone.utc)-timedelta(days=10)).isoformat()
   with p.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","ticker","lado","fecha","capturado_en","probabilidad_modelo"]); w.writeheader()
    for pid,ticker,side in [("yes","KXTEST-YES","YES"),("no","KXTEST-NO","NO"),("void","KXTEST-VOID","YES")]:
     w.writerow({"prediction_id":pid,"ticker":ticker,"lado":side,"fecha":old,"capturado_en":old,"probabilidad_modelo":"0.6"})
   fake={"KXTEST-YES":{"ticker":"KXTEST-YES","status":"settled","result":"yes"},"KXTEST-NO":{"ticker":"KXTEST-NO","status":"settled","result":"yes"},"KXTEST-VOID":{"ticker":"KXTEST-VOID","status":"settled","result":"void"}}
   cols=["prediction_id","estado","marcador","resuelto_en","resultado_kalshi"]
   with patch.object(core,"kalshi_get",side_effect=lambda path: {"market": fake[path.split("/",1)[1]]}): s=core.liquidate_kalshi_ledger(p,r,cols,side_field="lado")
   self.assertEqual((s["win"],s["loss"],s["anuladas"]),(1,1,1))
   with r.open(encoding="utf-8") as f: rows={x["prediction_id"]:x for x in csv.DictReader(f)}
   self.assertEqual(rows["yes"]["estado"],"WIN"); self.assertEqual(rows["no"]["estado"],"LOSS"); self.assertEqual(rows["void"]["estado"],"ANULADA")
   self.assertIn("anulado/cancelado", rows["void"]["marcador"])
   with patch.object(core,"_market_rows_for_series",return_value=fake): s=core.liquidate_kalshi_ledger(p,r,cols,side_field="lado")
   self.assertEqual(s["resultados_nuevos"],0)
 def test_rank_filters_duplicate_correlated_families_and_recomputes_edge(self):
  rows=[{"Ticker":"KXSOCCERTOTAL-1","Probabilidad del modelo":"80%","Precio de compra":"70%","Ventaja (edge)":-0.8,"Apuesta":"Más de 1.5 goles"},
   {"Ticker":"KXSOCCERTOTAL-2","Probabilidad del modelo":"75%","Precio de compra":"65%","Ventaja (edge)":-0.8,"Apuesta":"Más de 2.5 goles"},
   {"Ticker":"KXSOCCERBTTS-3","Probabilidad del modelo":"72%","Precio de compra":"60%","Ventaja (edge)":-0.8,"Apuesta":"Ambos marcan"},
   {"Ticker":"KXSOCCERCORNERS-4","Probabilidad del modelo":"70%","Precio de compra":"55%","Ventaja (edge)":-0.8,"Apuesta":"4+ corners"}]
  high,value=core.rank_market_candidates(rows,0.7)
  self.assertEqual(len(high),2); self.assertEqual(len(value),3)
  self.assertEqual(high[0]["Apuesta"],"Más de 1.5 goles")
  self.assertGreater(float(high[0]["Ventaja (edge)"]),0)

 def test_market_lookup_falls_back_to_ticker_filtered_list_endpoint(self):
  with tempfile.TemporaryDirectory() as d:
   p,r=Path(d)/"p.csv",Path(d)/"r.csv"
   old=(datetime.now(timezone.utc)-timedelta(days=5)).isoformat()
   with p.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","ticker","lado","fecha","capturado_en","probabilidad_modelo"]); w.writeheader()
    w.writerow({"prediction_id":"fallback","ticker":"KXTEST-FALLBACK","lado":"YES","fecha":old,"capturado_en":old,"probabilidad_modelo":"0.6"})
   def lookup(path, params=None):
    if path == "markets/KXTEST-FALLBACK":
     raise core.requests.HTTPError("direct endpoint unavailable")
    if path == "markets" and params and params.get("tickers") == "KXTEST-FALLBACK":
     return {"markets":[{"ticker":"KXTEST-FALLBACK","status":"settled","result":"yes"}]}
    raise AssertionError((path, params))
   with patch.object(core,"kalshi_get",side_effect=lookup):
    summary=core.liquidate_kalshi_ledger(p,r,["prediction_id","estado","marcador","resuelto_en"],side_field="lado")
   self.assertEqual(summary["win"],1)
   with r.open(encoding="utf-8") as f:
    rows=list(csv.DictReader(f))
   self.assertEqual(rows[0]["estado"],"WIN")

 def test_legacy_score_based_result_is_retried_against_kalshi(self):
  with tempfile.TemporaryDirectory() as d:
   p,r=Path(d)/"p.csv",Path(d)/"r.csv"
   old=(datetime.now(timezone.utc)-timedelta(days=5)).isoformat()
   with p.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","ticker","lado","fecha","capturado_en","probabilidad_modelo"]); w.writeheader()
    w.writerow({"prediction_id":"legacy","ticker":"KXTEST-REAL","lado":"YES","fecha":old,"capturado_en":old,"probabilidad_modelo":"0.6"})
   with r.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","estado","marcador","resuelto_en"]); w.writeheader()
    w.writerow({"prediction_id":"legacy","estado":"WIN","marcador":"3-0","resuelto_en":old})
   with patch.object(core,"kalshi_get",return_value={"market":{"ticker":"KXTEST-REAL","status":"settled","result":"no"}}):
    summary=core.liquidate_kalshi_ledger(p,r,["prediction_id","estado","marcador","resuelto_en"],side_field="lado")
   self.assertEqual(summary["loss"],1)
   with r.open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),1)
   self.assertEqual(rows[0]["estado"],"LOSS")
   self.assertEqual(rows[0]["marcador"],"Resultado oficial Kalshi: NO")


 def test_legacy_mlb_rows_without_match_or_probability_keep_existing_terminal_result(self):
  with tempfile.TemporaryDirectory() as d:
   p,r=Path(d)/"p.csv",Path(d)/"r.csv"
   with p.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","ticker","match_id","lado","fecha","capturado_en","probabilidad_modelo"]); w.writeheader()
    w.writerow({"prediction_id":"legacy","ticker":"KXMLBCHAMP-OLD","match_id":"","lado":"YES","probabilidad_modelo":"","fecha":"","capturado_en":""})
   with r.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","estado","resultado_kalshi","marcador","resuelto_en"]); w.writeheader()
    w.writerow({"prediction_id":"legacy","estado":"WIN","resultado_kalshi":"YES","marcador":"YES","resuelto_en":"2026-10-01T00:00:00+00:00"})
   with patch.object(core,"_market_rows_for_series",return_value={}) as fetch:
    summary=core.liquidate_kalshi_ledger(p,r,["prediction_id","estado","resultado_kalshi","marcador","resuelto_en"],side_field="lado")
   self.assertEqual(summary["pendientes_revisadas"],0)
   fetch.assert_not_called()
   with r.open(encoding="utf-8") as f:
    rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),1)
   self.assertEqual(rows[0]["estado"],"WIN")


 def test_line_extraction_and_csv_schema_migration(self):
  self.assertEqual(core.extract_line("Over 2.5 goals"),"2.5")
  self.assertEqual(core.extract_line("Handicap +1.5"),"+1.5")
  self.assertEqual(core.extract_line("3+ corners"),"3+")
  self.assertEqual(core.extract_line("Los Angeles D wins"),"")
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"pred.csv"
   with path.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","ticker","fecha","capturado_en","p"]); w.writeheader()
    w.writerow({"prediction_id":"original","ticker":"KXREAL-1","fecha":"2026-10-01T00:00:00+00:00","capturado_en":"2026-09-30T00:00:00+00:00","p":"0.75"})
   columns=["prediction_id","ticker","fecha","capturado_en","p","mercado","line","seleccion","bloque","edge"]
   merged=core.ensure_csv_schema(path,columns)
   with path.open(encoding="utf-8") as f:
    reader=csv.DictReader(f); rows=list(reader); header=reader.fieldnames
   self.assertEqual(header,columns)
   self.assertEqual(rows[0]["prediction_id"],"original")
   self.assertEqual(rows[0]["p"],"0.75")
   self.assertEqual(rows[0]["line"],"")

 def test_missing_capture_timestamp_is_written_and_old_schema_migrates(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"pred.csv"
   old_columns=["prediction_id","ticker","match_id","fecha","probabilidad_modelo"]
   with path.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=old_columns); w.writeheader()
    w.writerow({"prediction_id":"old","ticker":"KXREAL-OLD","match_id":"m1","fecha":"2026-10-20T00:00:00+00:00","probabilidad_modelo":"0.75"})
   cols=old_columns+["capturado_en","mercado","line","seleccion","bloque","edge"]
   future=(datetime.now(timezone.utc)+timedelta(days=2)).isoformat()
   new={"prediction_id":"new","ticker":"KXREAL-NEW","match_id":"m2","fecha":future,"probabilidad_modelo":"0.66","mercado":"Over 2.5 goals","line":"2.5","seleccion":"YES","bloque":"SENCILLAS"}
   summary=core.append_immutable_predictions(path,cols,[new],probability_field="probabilidad_modelo")
   self.assertEqual(summary["accepted"],1)
   with path.open(encoding="utf-8") as f:
    reader=csv.DictReader(f); rows=list(reader); header=reader.fieldnames
   self.assertEqual(header,cols)
   self.assertEqual(rows[0]["prediction_id"],"old")
   self.assertEqual(rows[1]["prediction_id"],"new")
   self.assertTrue(rows[1]["capturado_en"])
   self.assertLess(core.parse_datetime(rows[1]["capturado_en"]),core.parse_datetime(rows[1]["fecha"]))

 def test_legacy_no_evaluable_is_replaced_by_official_kalshi_result(self):
  with tempfile.TemporaryDirectory() as d:
   p,r=Path(d)/"p.csv",Path(d)/"r.csv"
   old=(datetime.now(timezone.utc)-timedelta(days=3)).isoformat()
   with p.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","ticker","match_id","lado","fecha","capturado_en","probabilidad_modelo"]); w.writeheader()
    w.writerow({"prediction_id":"old","ticker":"KXTEST-VOIDLEGACY","match_id":"m1","lado":"YES","fecha":old,"capturado_en":old,"probabilidad_modelo":"0.7"})
   with r.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","estado","marcador","resuelto_en"]); w.writeheader()
    w.writerow({"prediction_id":"old","estado":"NO_EVALUABLE","marcador":"No había estadística","resuelto_en":old})
   settled={"KXTEST-VOIDLEGACY":{"ticker":"KXTEST-VOIDLEGACY","status":"settled","result":"yes"}}
   with patch.object(core,"kalshi_get",return_value={"market":settled["KXTEST-VOIDLEGACY"]}):
    summary=core.liquidate_kalshi_ledger(p,r,["prediction_id","estado","marcador","resuelto_en"],side_field="lado")
   with r.open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
   self.assertEqual(summary["win"],1)
   self.assertEqual(len(rows),1)
   self.assertEqual(rows[0]["estado"],"WIN")
   self.assertNotEqual(rows[0]["estado"],"NO_EVALUABLE")
 def test_future_fixture_is_not_polled_before_kickoff(self):
  with tempfile.TemporaryDirectory() as d:
   p,r=Path(d)/"p.csv",Path(d)/"r.csv"
   future=(datetime.now(timezone.utc)+timedelta(days=2)).isoformat()
   with p.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","ticker","match_id","lado","fecha","capturado_en","probabilidad_modelo"]); w.writeheader()
    w.writerow({"prediction_id":"future","ticker":"KXTEST-FUTURE","match_id":"m1","lado":"YES","fecha":future,"capturado_en":datetime.now(timezone.utc).isoformat(),"probabilidad_modelo":"0.7"})
   with patch.object(core,"kalshi_get") as fetch, patch.object(core,"_market_rows_for_series") as catalog:
    summary=core.liquidate_kalshi_ledger(p,r,["prediction_id","estado","marcador","resuelto_en"],side_field="lado")
   self.assertEqual(summary["pendientes_revisadas"],0)
   fetch.assert_not_called()
   catalog.assert_not_called()



 def test_officially_postponed_future_match_is_annulled_without_kalshi_call(self):
  with tempfile.TemporaryDirectory() as tmp:
   p,r=Path(tmp)/"predicciones.csv",Path(tmp)/"resultados.csv"
   future=(datetime.now(timezone.utc)+timedelta(days=2)).isoformat()
   with p.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["prediction_id","ticker","match_id","lado","fecha","capturado_en","probabilidad_modelo"])
    w.writeheader()
    w.writerow({"prediction_id":"postponed","ticker":"KXTEST-GAME","match_id":"m-postponed","lado":"YES","fecha":future,"capturado_en":datetime.now(timezone.utc).isoformat(),"probabilidad_modelo":"0.7"})
   with patch.object(core,"kalshi_get") as fetch:
    summary=core.liquidate_kalshi_ledger(p,r,["prediction_id","estado","marcador","resuelto_en"],side_field="lado",cancelled_match_ids={"m-postponed"})
   self.assertEqual(summary["anuladas"],1)
   fetch.assert_not_called()
   with r.open(newline="",encoding="utf-8") as f:
    row=next(csv.DictReader(f))
   self.assertEqual(row["estado"],"ANULADA")
   self.assertIn("fuente oficial",row["marcador"])

 def test_single_candidates_filter_trivial_price_and_extreme_edges(self):
  rows=[
   {"Ticker":"KXGOALS-TOTAL-OVER05","Apuesta sencilla":"Más de 0.5 goles","Categoría resumen":"Goles totales","Probabilidad del modelo":"97%","Precio de compra":"96%","Ventaja (edge)":0.01,"line":"0.5"},
   {"Ticker":"KXGOALS-TOTAL-OVER25","Apuesta sencilla":"Más de 2.5 goles","Categoría resumen":"Goles totales","Probabilidad del modelo":"72%","Precio de compra":"60%","Ventaja (edge)":0.12,"line":"2.5"},
   {"Ticker":"KXCORNERS-TOTAL-OVER85","Apuesta sencilla":"Más de 8.5 córners","Categoría resumen":"Corners totales","Probabilidad del modelo":"70%","Precio de compra":"55%","Ventaja (edge)":0.15,"line":"8.5"},
   {"Ticker":"KXBTTS-YES","Apuesta sencilla":"Ambos equipos marcan","Categoría resumen":"Ambos marcan","Probabilidad del modelo":"74%","Precio de compra":"50%","Ventaja (edge)":0.24},
   {"Ticker":"KXGOALS-TOTAL-OVER35","Apuesta sencilla":"Más de 3.5 goles","Categoría resumen":"Goles totales","Probabilidad del modelo":"60%","Precio de compra":"45%","Ventaja (edge)":0.15,"line":"3.5"},
   {"Ticker":"KXPRICE-HIGH","Apuesta sencilla":"Gana local","Categoría resumen":"Resultado","Probabilidad del modelo":"95%","Precio de compra":"91%","Ventaja (edge)":0.04},
  ]
  selected=core.rank_single_candidates(rows)
  tickers=[x["Ticker"] for x in selected]
  self.assertNotIn("KXGOALS-TOTAL-OVER05",tickers)
  self.assertNotIn("KXPRICE-HIGH",tickers)
  self.assertNotIn("KXBTTS-YES",tickers)  # >20 pp apparent edge requires review
  self.assertIn("KXCORNERS-TOTAL-OVER85",tickers)
  self.assertEqual(tickers.count("KXGOALS-TOTAL-OVER25")+tickers.count("KXGOALS-TOTAL-OVER35"),1)

 def test_csv_reader_quarantines_bad_row_and_keeps_good_history(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/"resultados.csv"
   path.write_text("\n".join([
    "prediction_id,estado,marcador,resuelto_en",
    "old,WIN,Marcador,2026-10-10T10:00:00Z",
    "bad,LOSS,Marcador,extra,more,bad-time",
    "new,LOSS,Marcador,2026-10-10T12:00:00Z",
   ]) + "\n",encoding="utf-8")
   rows=core.read_csv_rows_resilient(path)
   self.assertEqual([row["prediction_id"] for row in rows],["old","new"])
   self.assertNotIn("bad,LOSS,Marcador,extra",path.read_text(encoding="utf-8"))
   quarantine=path.with_suffix(path.suffix+".malformed_rows.csv")
   self.assertTrue(quarantine.exists())
   self.assertIn("bad,LOSS,Marcador,extra",quarantine.read_text(encoding="utf-8"))

 def test_csv_reader_migrates_legacy_football_rows_without_losing_capture_time(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/"predicciones_laliga.csv"
   current=["prediction_id","match_id","fecha","local","visita","categoria","jugada","probabilidad_base","probabilidad_calibrada","calibracion","precio","disponibilidad","ticker","mercado","line","seleccion","bloque","edge","clave_interna","capturado_en"]
   legacy=["prediction_id","match_id","fecha","local","visita","categoria","jugada","probabilidad_base","probabilidad_calibrada","calibracion","precio","disponibilidad","ticker","clave_interna","capturado_en"]
   old=["old-id","m1","2026-10-20T18:00:00Z","A","B","Goles totales","Más de 2.5 goles","0.7","0.7","","40%","","KXREAL-1","(('KXLALIGATOTAL', 'over', 2.5, 'over'), 'YES')","2026-10-10T12:00:00Z"]
   with path.open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(current); w.writerow(old)
   rows=core.read_csv_rows_resilient(path)
   self.assertEqual(len(rows),1)
   self.assertEqual(rows[0]["prediction_id"],"old-id")
   self.assertEqual(rows[0]["capturado_en"],"2026-10-10T12:00:00Z")
   self.assertEqual(rows[0]["line"],"2.5")
   with path.open(newline="",encoding="utf-8") as f:
    saved=list(csv.DictReader(f))
   self.assertEqual(len(saved),1)
   self.assertEqual(len(saved[0]),len(current))

 def test_single_candidates_reject_missing_price_and_negative_edge(self):
  rows=[
   {"Ticker":"KX-NOPRICE","Apuesta sencilla":"Ambos marcan","Categoría resumen":"Ambos marcan","Probabilidad del modelo":"70%","Precio de compra":"—","Ventaja (edge)":0.1},
   {"Ticker":"KX-NEGATIVE","Apuesta sencilla":"Ambos marcan","Categoría resumen":"Ambos marcan","Probabilidad del modelo":"60%","Precio de compra":"65%","Ventaja (edge)":-0.05},
  ]
  self.assertEqual(core.rank_single_candidates(rows),[])

 def test_combo_and_value_require_positive_net_edge_and_price_caps(self):
  rows=[
   {"Ticker":"KXGAME-EXPENSIVE","Apuesta":"Gana local","Probabilidad del modelo":"97%","Precio de compra":"96%","Ventaja (edge)":0.50},
   {"Ticker":"KXBTTS-AFFORDABLE","Apuesta":"Ambos equipos marcan","Probabilidad del modelo":"85%","Precio de compra":"78%","Ventaja (edge)":-0.50},
   {"Ticker":"KXGOALS-NO-PRICE","Apuesta":"Más de 2.5 goles","Probabilidad del modelo":"80%","Precio de compra":"—","Ventaja (edge)":0.50},
  ]
  high,value=core.rank_market_candidates(rows,min_probability=0.65)
  self.assertEqual([x["Ticker"] for x in high],["KXBTTS-AFFORDABLE"])
  self.assertEqual([x["Ticker"] for x in value],["KXBTTS-AFFORDABLE"])
 def test_market_ranking_uses_calibrated_probability_and_rejects_extreme_edge(self):
  rows=[
   {"Ticker":"KXGAME-CAL","Apuesta":"Gana local","Probabilidad del modelo":"90%","probabilidad_calibrada":"60%","Precio de compra":"50%"},
   {"Ticker":"KXGAME-EXTREME","Apuesta":"Gana visitante","Probabilidad del modelo":"90%","Precio de compra":"50%"},
  ]
  high,value=core.rank_market_candidates(rows,min_probability=0.55)
  self.assertEqual([x["Ticker"] for x in high],["KXGAME-CAL"])
  self.assertEqual([x["Ticker"] for x in value],["KXGAME-CAL"])


if __name__=="__main__": unittest.main()
