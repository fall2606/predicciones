import requests, pandas as pd, numpy as np, streamlit as st
from app_metadata import mostrar_metadata
from scipy.stats import poisson
from football_match_utils import is_upcoming_match

st.set_page_config(page_title="Premier League · pronósticos", layout="wide")
st.title("⚽ Premier League · pronósticos del modelo")
mostrar_metadata()
st.page_link("pages/Contexto_deportivo.py", label="Analizar noticias, lesiones y contexto antes de decidir", icon="🧠")

CLAVE = st.secrets["CLAVE"]
URL = "https://api.football-data.org/v4/competitions/PL/matches"
K = 6

def pedir(**params):
    r = requests.get(URL, headers={"X-Auth-Token": CLAVE}, params=params)
    r.raise_for_status()
    return r.json()["matches"]

@st.cache_data(ttl=300)
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
    prox = [m for m in actual if is_upcoming_match(m)]
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
if tabla.empty:
    st.info("No hay partidos próximos disponibles.")
else:
    for fila in tabla.to_dict("records"):
        with st.container(border=True):
            st.markdown(f"**{fila.get('Partido', 'Partido')}** · {fila.get('Fecha', '')}")
            st.write(f"Marcador estimado: **{fila.get('Goles esp.', '—')}**")
            cols = st.columns(4)
            for col, etiqueta in zip(cols, ("Gana local %", "Empate %", "Gana visita %", "Over 2.5 %")):
                col.metric(etiqueta.replace(" %", ""), f"{fila.get(etiqueta, '—')}%")
            st.caption("Estimación estadística básica; no es una apuesta y aún no está validada con una muestra histórica suficiente.")
st.warning("Modelo estadístico básico. No se recomienda apostar con este modelo hasta medir su rendimiento fuera de muestra.")
