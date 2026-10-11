"""Backtest MLB de probabilidades guardadas contra el precio Kalshi."""
import csv, math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
PRED=ROOT/"predicciones_mlb.csv"; RES=ROOT/"resultados_mlb.csv"; OUT=ROOT/"backtest_mlb_resumen.csv"
FIELDS=["tipo_mercado","n_resueltos","n_con_precio","brier_modelo","brier_precio_kalshi","diferencia_modelo_menos_precio","estado_calibracion"]
def read(path):
    if not path.exists() or not path.stat().st_size: return []
    with path.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))
def prob(x):
    try: p=float(x)
    except (TypeError,ValueError,OverflowError): return None
    return p if math.isfinite(p) and 0<=p<=1 else None
def kind(r):
    s=" ".join(str(r.get(k,"")) for k in ("categoria","mercado","ticker")).casefold()
    if "inningwin" in s or "primera entrada" in s or "first inning" in s: return "primera_entrada"
    if any(x in s for x in ("strikeout","home run","pitcher","batter")): return "props_jugador"
    if "run line" in s or "spread" in s or "margen" in s: return "run_line"
    if "team total" in s or "total por equipo" in s: return "total_equipo"
    if any(x in s for x in ("over","under","total de carreras","total runs")): return "total_carreras"
    if any(x in s for x in ("winner","moneyline","ganador","game")): return "ganador"
    return str(r.get("categoria") or "otro")
def main():
    preds=read(PRED); results={str(r.get("prediction_id","")).strip():r for r in read(RES)}
    groups=defaultdict(list)
    for p in preds:
        r=results.get(str(p.get("prediction_id","")).strip())
        if not r or str(r.get("estado","")).upper() not in {"WIN","LOSS"}: continue
        y=1.0 if str(r.get("estado","")).upper()=="WIN" else 0.0
        mp=prob(p.get("probabilidad_modelo")); price=prob(p.get("precio_captura"))
        if mp is not None: groups[kind(p)].append((mp,price,y))
    rows=[]
    types=sorted(set(groups)|{"primera_entrada","total_carreras","total_equipo","run_line","ganador","props_jugador"})
    for t in types:
        vals=groups.get(t,[]); n_resueltos=len(vals)
        common=[(p,q,y) for p,q,y in vals if q is not None]
        n=len(common)
        bm=sum((p-y)**2 for p,_,y in common)/n if n else None
        bk=sum((q-y)**2 for _,q,y in common)/n if n else None
        rows.append({"tipo_mercado":t,"n_resueltos":n_resueltos,"n_con_precio":n,"brier_modelo":"" if bm is None else f"{bm:.6f}","brier_precio_kalshi":"" if bk is None else f"{bk:.6f}","diferencia_modelo_menos_precio":"" if bm is None or bk is None else f"{bm-bk:.6f}","estado_calibracion":"CALIBRACIÓN AÚN INSUFICIENTE" if n<30 else "MUESTRA PARA EVALUAR"})
    with OUT.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"BACKTEST MLB: contratos resueltos con probabilidad válida={sum(len(v) for v in groups.values())}; informe={OUT.name}")
    for r in rows: print(f"BACKTEST MLB · {r['tipo_mercado']}: resueltos={r['n_resueltos']} · comparables={r['n_con_precio']} · Brier modelo={r['brier_modelo'] or 'N/D'} · Brier precio Kalshi={r['brier_precio_kalshi'] or 'N/D'} · {r['estado_calibracion']}")
if __name__=="__main__": main()
