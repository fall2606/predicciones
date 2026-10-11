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
   with patch.object(core,"_market_rows_for_series",return_value=fake): s=core.liquidate_kalshi_ledger(p,r,cols,side_field="lado")
   self.assertEqual((s["win"],s["loss"],s["anuladas"]),(1,1,1))
   with r.open(encoding="utf-8") as f: rows={x["prediction_id"]:x for x in csv.DictReader(f)}
   self.assertEqual(rows["yes"]["estado"],"WIN"); self.assertEqual(rows["no"]["estado"],"LOSS"); self.assertEqual(rows["void"]["estado"],"ANULADA")
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
   with patch.object(core,"_market_rows_for_series",return_value={"KXTEST-REAL":{"ticker":"KXTEST-REAL","status":"settled","result":"no"}}):
    summary=core.liquidate_kalshi_ledger(p,r,["prediction_id","estado","marcador","resuelto_en"],side_field="lado")
   self.assertEqual(summary["loss"],1)
   with r.open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),1)
   self.assertEqual(rows[0]["estado"],"LOSS")
   self.assertEqual(rows[0]["marcador"],"Resultado oficial Kalshi: NO")

if __name__=="__main__": unittest.main()
