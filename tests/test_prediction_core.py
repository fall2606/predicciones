import csv, tempfile, unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
import prediction_core as core

class CoreTests(unittest.TestCase):
 def test_probability_validation(self):
  for v in ("NaN","inf","-inf",-0.1,1.1,None,""): self.assertIsNone(core.valid_probability(v))
  self.assertEqual(core.valid_probability(0),0); self.assertEqual(core.valid_probability(1),1)
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
 def test_rank_filters_duplicate_correlated_families(self):
  rows=[{"Ticker":"KXSOCCERTOTAL-1","Probabilidad del modelo":"90%","Ventaja (edge)":0.1,"Apuesta":"Más de 1.5 goles"},
   {"Ticker":"KXSOCCERTOTAL-2","Probabilidad del modelo":"85%","Ventaja (edge)":0.08,"Apuesta":"Más de 2.5 goles"},
   {"Ticker":"KXSOCCERBTTS-3","Probabilidad del modelo":"80%","Ventaja (edge)":0.07,"Apuesta":"Ambos marcan"},
   {"Ticker":"KXSOCCERCORNERS-4","Probabilidad del modelo":"78%","Ventaja (edge)":0.05,"Apuesta":"4+ corners"}]
  high,value=core.rank_market_candidates(rows,0.7)
  self.assertEqual(len(high),2); self.assertEqual(len(value),3); self.assertEqual(high[0]["Apuesta"],"Más de 1.5 goles")

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


 def test_legacy_mlb_rows_without_match_or_probability_are_excluded(self):
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
   fetch.assert_called_once_with(set())
   with r.open(encoding="utf-8") as f: self.assertEqual(list(csv.DictReader(f)),[])


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

if __name__=="__main__": unittest.main()
