import os, difflib, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from scipy.stats import poisson, nbinom

CLAVE = os.environ["CLAVE"]
URL = "https://api.football-data.org/v4/competitions/PL/matches"
K = 6
N = 12
NC = 40
R = 16
DIAS = 7
ARCHIVO = "predicciones.csv"

# Mercados de goles: nombre -> condición sobre i (goles local) y j (goles visita)
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

# Mercados de corners: i = corners local, j = corners visita
MC = {}
for k in range(1, 13):
    MC[f"Corners local: {k}+"] = lambda i, j, k=k: i >= k
    MC[f"Corners visita: {k}+"] = lambda i, j, k=k: j >= k
for x in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]:
    MC[f"Más de {x} corners totales"] = lambda i, j, x=x: (i + j) > x

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

# ---- Modelo de goles ----
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

# ---- Modelo de corners (datos de football-data.co.uk) ----
partes = []
for temp in ["2526", "2627"]:
    try:
        x = pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{temp}/E0.csv",
                        encoding_errors="ignore", on_bad_lines="skip")
        x = x[["HomeTeam", "AwayTeam", "HC", "AC"]].dropna().copy()
        x["temp"] = temp
        partes.append(x)
    except Exception as e:
        print("Aviso: no se pudieron bajar corners de", temp, e)
hay_corners = len(partes) > 0
if hay_corners:
    corners = pd.concat(partes, ignore_index=True)
    tc = pd.concat([
        pd.DataFrame({"equipo": corners["HomeTeam"], "cf": corners["HC"], "cc": corners["AC"]}),
        pd.DataFrame({"equipo": corners["AwayTeam"], "cf": corners["AC"], "cc": corners["HC"]}),
    ])
    hc_prom, ac_prom = corners["HC"].mean(), corners["AC"].mean()
    promc = (hc_prom + ac_prom) / 2
    gc = tc.groupby("equipo").agg(cf=("cf", "sum"), cc=("cc", "sum"), n=("cf", "count"))
    ataque_c = ((gc["cf"] + K * promc) / (gc["n"] + K)) / promc
    defensa_c = ((gc["cc"] + K * promc) / (gc["n"] + K)) / promc
    nombres_c = list(gc.index)
    reales_c = {(r.HomeTeam, r.AwayTeam): (r.HC, r.AC)
                for r in corners[corners["temp"] == "2627"].itertuples()}

ALIAS = {"nottingham": "Nott'm Forest", "brighton hove": "Brighton",
         "leeds united": "Leeds", "wolverhampton": "Wolves",
         "coventry city": "Coventry", "hull city": "Hull",
         "ipswich town": "Ipswich"}

def nombre_c(n):
    if not hay_corners:
        return None
    if n in nombres_c:
        return n
    a = ALIAS.get(n.lower())
    if a in nombres_c:
        return a
    cerca = difflib.get_close_matches(n, nombres_c, n=1, cutoff=0.6)
    if cerca:
        return cerca[0]
    print("Aviso: sin datos de corners para", n)
    return None

# ---- Historial existente ----
cols = ["id", "fecha", "local", "visita", "mercado", "p", "gano",
        "goles_local", "goles_visita"]
hist = pd.read_csv(ARCHIVO) if os.path.exists(ARCHIVO) else pd.DataFrame(columns=cols)
ya_g = set(hist[hist["mercado"].isin(M)]["id"].astype(int))
ya_c = set(hist[hist["mercado"].isin(MC)]["id"].astype(int))

# ---- Predicciones nuevas ----
I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
IC, JC = np.meshgrid(np.arange(NC), np.arange(NC), indexing="ij")
ahora = datetime.now(timezone.utc)
limite = ahora + timedelta(days=DIAS)
nuevas = []
for m in actual:
    if m["status"] not in ("SCHEDULED", "TIMED"):
        continue
    inicio = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
    if not (ahora < inicio <= limite):
        continue
    quiere_g = m["id"] not in ya_g
    quiere_c = hay_corners and m["id"] not in ya_c
    if not (quiere_g or quiere_c):
        continue
    l, v = m["homeTeam"]["shortName"], m["awayTeam"]["shortName"]
    base = {"id": m["id"], "fecha": m["utcDate"][:10], "local": l, "visita": v,
            "gano": np.nan, "goles_local": np.nan, "goles_visita": np.nan}
    if quiere_g:
        gl = pl * ataque.get(l, 1.0) * defensa.get(v, 1.0)
        gv = pv * ataque.get(v, 1.0) * defensa.get(l, 1.0)
        p = np.outer(poisson.pmf(np.arange(N), gl), poisson.pmf(np.arange(N), gv))
        for nombre, f in M.items():
            nuevas.append({**base, "mercado": nombre, "p": round(float(p[f(I, J)].sum()), 4)})
        total = gl + gv
        for nombre, prob in [("Primer gol: local", gl / total * (1 - np.exp(-total))),
                             ("Primer gol: visita", gv / total * (1 - np.exp(-total))),
                             ("Primer gol: ninguno", np.exp(-total))]:
            nuevas.append({**base, "mercado": nombre, "p": round(float(prob), 4)})
    if quiere_c:
        cl, cv = nombre_c(l), nombre_c(v)
        ml = hc_prom * ataque_c.get(cl, 1.0) * defensa_c.get(cv, 1.0)
        mv = ac_prom * ataque_c.get(cv, 1.0) * defensa_c.get(cl, 1.0)
        pc = np.outer(nbinom.pmf(np.arange(NC), R, R / (R + ml)),
                      nbinom.pmf(np.arange(NC), R, R / (R + mv)))
        for nombre, f in MC.items():
            nuevas.append({**base, "mercado": nombre, "p": round(float(pc[f(IC, JC)].sum()), 4)})

partidos_nuevos = 0
if nuevas:
    nuevas = pd.DataFrame(nuevas)
    partidos_nuevos = nuevas["id"].nunique()
    hist = nuevas if hist.empty else pd.concat([hist, nuevas], ignore_index=True)

# ---- Anotar resultados de partidos terminados ----
terminados = {m["id"]: m["score"]["fullTime"] for m in actual
              if m["status"] == "FINISHED" and m["score"]["fullTime"]["home"] is not None}
for k in hist.index[hist["gano"].isna()]:
    idp = int(hist.at[k, "id"])
    nombre = hist.at[k, "mercado"]
    if nombre in M and idp in terminados:
        gl_r, gv_r = terminados[idp]["home"], terminados[idp]["away"]
        hist.at[k, "gano"] = int(bool(M[nombre](gl_r, gv_r)))
        hist.at[k, "goles_local"] = gl_r
        hist.at[k, "goles_visita"] = gv_r
    elif nombre in MC and hay_corners and idp in terminados:
        clave = (nombre_c(hist.at[k, "local"]), nombre_c(hist.at[k, "visita"]))
        if clave in reales_c:
            c1, c2 = reales_c[clave]
            hist.at[k, "gano"] = int(bool(MC[nombre](c1, c2)))

hist.to_csv(ARCHIVO, index=False)
print("Partidos con filas nuevas:", partidos_nuevos, "| Filas en el historial:", len(hist))
