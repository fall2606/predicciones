import os, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from scipy.stats import poisson

CLAVE = os.environ["CLAVE"]
URL = "https://api.football-data.org/v4/competitions/PL/matches"
K = 6
N = 12
ARCHIVO = "predicciones.csv"

# Mercados: nombre -> condición sobre i (goles local) y j (goles visita)
M = {
    "Gana local": lambda i, j: i > j,
    "Empate": lambda i, j: i == j,
    "Gana visita": lambda i, j: i < j,
    "Ambos marcan": lambda i, j: (i > 0) & (j > 0),
}
for x in [1.5, 2.5, 3.5]:
    M[f"Local gana por más de {x}"] = lambda i, j, x=x: (i - j) > x
    M[f"Visita gana por más de {x}"] = lambda i, j, x=x: (j - i) > x
for x in [1.5, 2.5, 3.5, 4.5, 5.5]:
    M[f"Más de {x} goles"] = lambda i, j, x=x: (i + j) > x
for x in [0.5, 1.5, 2.5, 3.5]:
    M[f"Local más de {x} goles"] = lambda i, j, x=x: i > x
    M[f"Visita más de {x} goles"] = lambda i, j, x=x: j > x
for a in range(5):
    for b in range(5):
        M[f"Marcador {a}-{b}"] = lambda i, j, a=a, b=b: (i == a) & (j == b)

def pedir(**params):
    r = requests.get(URL, headers={"X-Auth-Token": CLAVE}, params=params)
    r.raise_for_status()
    return r.json()["matches"]

actual = pedir()
todos = list(actual)
try:
    todos += pedir(season=2025)
except Exception:
    pass

# Modelo: fuerza de ataque y defensa, mezclada con el promedio de la liga
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

# Historial existente
cols = ["id", "fecha", "local", "visita", "mercado", "p", "gano",
        "goles_local", "goles_visita"]
hist = pd.read_csv(ARCHIVO) if os.path.exists(ARCHIVO) else pd.DataFrame(columns=cols)
ya = set(hist["id"].astype(int))

# Predicciones nuevas (partidos de los próximos 2 días)
I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
ahora = datetime.now(timezone.utc)
limite = ahora + timedelta(days=2)
nuevas = []
for m in actual:
    if m["status"] not in ("SCHEDULED", "TIMED") or m["id"] in ya:
        continue
    inicio = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
    if not (ahora < inicio <= limite):
        continue
    l, v = m["homeTeam"]["shortName"], m["awayTeam"]["shortName"]
    gl = pl * ataque.get(l, 1.0) * defensa.get(v, 1.0)
    gv = pv * ataque.get(v, 1.0) * defensa.get(l, 1.0)
    p = np.outer(poisson.pmf(np.arange(N), gl), poisson.pmf(np.arange(N), gv))
    base = {"id": m["id"], "fecha": m["utcDate"][:10], "local": l, "visita": v,
            "gano": np.nan, "goles_local": np.nan, "goles_visita": np.nan}
    for nombre, f in M.items():
        nuevas.append({**base, "mercado": nombre, "p": round(float(p[f(I, J)].sum()), 4)})
    total = gl + gv
    for nombre, prob in [("Primer gol: local", gl / total * (1 - np.exp(-total))),
                         ("Primer gol: visita", gv / total * (1 - np.exp(-total))),
                         ("Primer gol: ninguno", np.exp(-total))]:
        nuevas.append({**base, "mercado": nombre, "p": round(float(prob), 4)})

partidos_nuevos = 0
if nuevas:
    nuevas = pd.DataFrame(nuevas)
    partidos_nuevos = nuevas["id"].nunique()
    hist = nuevas if hist.empty else pd.concat([hist, nuevas], ignore_index=True)

# Anotar resultados de partidos ya terminados
terminados = {m["id"]: m["score"]["fullTime"] for m in actual
              if m["status"] == "FINISHED" and m["score"]["fullTime"]["home"] is not None}
for k in hist.index[hist["gano"].isna()]:
    idp = int(hist.at[k, "id"])
    nombre = hist.at[k, "mercado"]
    if idp in terminados and nombre in M:
        gl_r, gv_r = terminados[idp]["home"], terminados[idp]["away"]
        hist.at[k, "gano"] = int(bool(M[nombre](gl_r, gv_r)))
        hist.at[k, "goles_local"] = gl_r
        hist.at[k, "goles_visita"] = gv_r

hist.to_csv(ARCHIVO, index=False)
print("Partidos nuevos:", partidos_nuevos, "| Filas en el historial:", len(hist))
