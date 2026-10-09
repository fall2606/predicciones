import requests, pandas as pd, numpy as np, streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title="Predicciones Premier League", layout="wide")
st.title("⚽ Predicciones Premier League")

CLAVE = st.secrets["CLAVE"]
URL = "https://api.football-data.org/v4/competitions/PL/matches"
K = 6

def pedir(**params):
    r = requests.get(URL, headers={"X-Auth-Token": CLAVE}, params=params)
    r.raise_for_status()
    return r.json()["matches"]

@st.cache_data(ttl=3600)
def calcular(n):
    actual = pedir()
    todos = list(actual)
    anterior = True
    try:
        todos += pedir(season=2025)
    except Exception:
        anterior = False
    filas = []
    for m in todos:
        g = m["score"]["fullTime"]
        if m["status"] == "FINISHED" and g["home"] is not None:
            l, v = m["homeTeam"]["shortName"], m["awayTeam"]["shortName"]
            filas.append((l, g["home"], g["away"]))
            filas.append((v, g["away"], g["home"]))
    t = pd.DataFrame(filas, columns=["equipo", "gf", "gc"])
    pl = t.iloc[0::2]["gf"].mean()
    pv = t.iloc[1::2]["gf"].mean()
    prom = (pl + pv) / 2
    g = t.groupby("equipo").agg(gf=("gf", "sum"), gc=("gc", "sum"), n=("gf", "count"))
    ataque = ((g["gf"] + K * prom) / (g["n"] + K)) / prom
    defensa = ((g["gc"] + K * prom) / (g["n"] + K)) / prom
    prox = [m for m in actual if m["status"] in ("SCHEDULED", "TIMED")]
    prox = sorted(prox, key=lambda m: m["utcDate"])[:n]
    suma = np.add.outer(np.arange(10), np.arange(10))
    res = []
    for m in prox:
        l, v = m["homeTeam"]["shortName"], m["awayTeam"]["shortName"]
        gl = pl * ataque.get(l, 1.0) * defensa.get(v, 1.0)
        gv = pv * ataque.get(v, 1.0) * defensa.get(l, 1.0)
        p = np.outer(poisson.pmf(np.arange(10), gl), poisson.pmf(np.arange(10), gv))
        over = p[suma > 2].sum()
        res.append({
            "Fecha": m["utcDate"][:10], "Partido": f"{l} vs {v}",
            "Goles esp.": f"{gl:.1f}-{gv:.1f}",
            "Gana local %": round(np.tril(p, -1).sum() * 100),
            "Empate %": round(np.trace(p) * 100),
            "Gana visita %": round(np.triu(p, 1).sum() * 100),
            "Over 2.5 %": round(over * 100),
            "Cuota justa over 2.5": round(1 / over, 2),
        })
    return pd.DataFrame(res), len(t) // 2, anterior

n = st.slider("Partidos a mostrar", 5, 20, 10)
tabla, usados, anterior = calcular(n)
st.caption(f"Partidos usados para el modelo: {usados} · Temporada anterior: {'sí' if anterior else 'no'}")
st.dataframe(tabla, use_container_width=True, hide_index=True)
st.warning("Modelo básico para practicar. No apuestes dinero real con esto todavía.")
