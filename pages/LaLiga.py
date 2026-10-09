from datetime import datetime, timedelta
from difflib import get_close_matches
from itertools import combinations
from unicodedata import normalize
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import nbinom, poisson


API_FD = "https://api.football-data.org/v4"
K = 6
N_GOLES = 11
N_CORNERS = 40
DISPERSION_CORNERS = 16
TZ = ZoneInfo("America/Chicago")

st.set_page_config(page_title="LaLiga · predicciones", page_icon="⚽", layout="wide")
st.title("🇪🇸 LaLiga · mercados y combinadas por partido")
st.caption(
    "Solo partidos de hoy y los próximos dos días. Elige un encuentro para ver el resumen "
    "de combinada y los mercados modelados."
)
st.link_button("Ver mercados actuales de LaLiga en Kalshi", "https://kalshi.com/combos/soccer/la-liga")


def pedir_fd(path, clave, **params):
    response = requests.get(
        f"{API_FD}/{path}",
        headers={"X-Auth-Token": clave},
        params=params,
        timeout=25,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=1800, show_spinner=False)
def cargar_partidos_y_estadisticas(clave, temporada, temporada_anterior):
    actual = pedir_fd("competitions/PD/matches", clave, season=temporada)["matches"]
    anteriores = []
    try:
        anteriores = pedir_fd(
            "competitions/PD/matches", clave, season=temporada_anterior
        )["matches"]
    except requests.RequestException:
        pass
    historial = [m for m in anteriores + actual
                 if m.get("status") == "FINISHED"
                 and m.get("score", {}).get("fullTime", {}).get("home") is not None]
    if not historial:
        raise ValueError("La API no devolvió partidos terminados para calcular el modelo.")

    home_rows, away_rows = [], []
    for m in historial:
        h = m["homeTeam"].get("shortName") or m["homeTeam"]["name"]
        a = m["awayTeam"].get("shortName") or m["awayTeam"]["name"]
        gh = int(m["score"]["fullTime"]["home"])
        ga = int(m["score"]["fullTime"]["away"])
        home_rows.append((h, gh, ga))
        away_rows.append((a, ga, gh))

    home_df = pd.DataFrame(home_rows, columns=["equipo", "gf", "gc"])
    away_df = pd.DataFrame(away_rows, columns=["equipo", "gf", "gc"])
    prom_home = float(home_df["gf"].mean())
    prom_away = float(away_df["gf"].mean())
    if prom_home <= 0 or prom_away <= 0:
        raise ValueError("Faltan goles suficientes en el historial de LaLiga.")

    home_stats = home_df.groupby("equipo").agg(gf=("gf", "sum"), gc=("gc", "sum"), n=("gf", "count"))
    away_stats = away_df.groupby("equipo").agg(gf=("gf", "sum"), gc=("gc", "sum"), n=("gf", "count"))
    home_attack = ((home_stats.gf + K * prom_home) / (home_stats.n + K)) / prom_home
    home_defence = ((home_stats.gc + K * prom_away) / (home_stats.n + K)) / prom_away
    away_attack = ((away_stats.gf + K * prom_away) / (away_stats.n + K)) / prom_away
    away_defence = ((away_stats.gc + K * prom_home) / (away_stats.n + K)) / prom_home
    return actual, prom_home, prom_away, home_attack.to_dict(), home_defence.to_dict(), away_attack.to_dict(), away_defence.to_dict()


def limpiar(nombre):
    return " ".join(normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode().lower().split())


@st.cache_data(ttl=21600, show_spinner=False)
def cargar_corners(temporadas):
    partes = []
    for temporada in temporadas:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{temporada}/SP1.csv"
            df = pd.read_csv(url, encoding_errors="ignore", on_bad_lines="skip")
            columnas = {c.lower(): c for c in df.columns}
            requeridas = [columnas.get(x) for x in ("hometeam", "awayteam", "hc", "ac")]
            if all(requeridas):
                df = df[requeridas].copy()
                df.columns = ["HomeTeam", "AwayTeam", "HC", "AC"]
                df = df.dropna()
                if not df.empty:
                    df["temporada"] = temporada
                    partes.append(df)
        except Exception:
            continue
    if not partes:
        return None

    partidos = pd.concat(partes, ignore_index=True)
    equipos = {}
    for r in partidos.itertuples(index=False):
        h, a = limpiar(r.HomeTeam), limpiar(r.AwayTeam)
        equipos.setdefault(h, {"home_for": [], "home_against": [], "away_for": [], "away_against": []})
        equipos.setdefault(a, {"home_for": [], "home_against": [], "away_for": [], "away_against": []})
        equipos[h]["home_for"].append(float(r.HC))
        equipos[h]["home_against"].append(float(r.AC))
        equipos[a]["away_for"].append(float(r.AC))
        equipos[a]["away_against"].append(float(r.HC))
    actuales = partidos[partidos["temporada"] == temporadas[-1]]
    resultados = {
        (limpiar(r.HomeTeam), limpiar(r.AwayTeam)): (float(r.HC), float(r.AC))
        for r in actuales.itertuples(index=False)
        if pd.notna(r.HC) and pd.notna(r.AC)
    }
    return {
        "equipos": equipos,
        "prom_home": float(partidos.HC.mean()),
        "prom_away": float(partidos.AC.mean()),
        "nombres": list(equipos),
        "resultados": resultados,
    }


def clave_equipo(nombre, nombres):
    exacta = limpiar(nombre)
    if exacta in nombres:
        return exacta
    match = get_close_matches(exacta, nombres, n=1, cutoff=0.58)
    return match[0] if match else None


def estimar_goles(local, visita, modelo):
    _, ph, pa, ha, hd, aa, ad = modelo
    lh = ph * ha.get(local, 1.0) * ad.get(visita, 1.0)
    av = pa * aa.get(visita, 1.0) * hd.get(local, 1.0)
    return max(0.15, lh), max(0.15, av)


def matriz_corner(media):
    valores = np.arange(N_CORNERS)
    p = np.outer(
        nbinom.pmf(valores, DISPERSION_CORNERS, DISPERSION_CORNERS / (DISPERSION_CORNERS + media[0])),
        nbinom.pmf(valores, DISPERSION_CORNERS, DISPERSION_CORNERS / (DISPERSION_CORNERS + media[1])),
    )
    return p / p.sum(), np.meshgrid(valores, valores, indexing="ij")


def calcular_mercados(p, i, j, local, visita, resultado=None, p_corners=None, ci=None, cj=None, corners_final=None):
    filas = []

    def agregar(nombre, grupo, evento, condicion, probabilidad=None):
        matriz = p_corners if evento == "corners" else p
        prob = float(probabilidad if probabilidad is not None else matriz[condicion].sum())
        acierto = None
        if resultado is not None:
            goles_l, goles_v = resultado
            if evento == "goles":
                acierto = bool(condicion[goles_l, goles_v])
            elif evento == "primer gol":
                acierto = None
            elif evento == "corners" and corners_final is not None and condicion is not None:
                acierto = bool(condicion[int(corners_final[0]), int(corners_final[1])])
        filas.append({"mercado": nombre, "grupo": grupo, "evento": evento,
                      "condicion": condicion, "p": prob, "acierto": acierto})

    agregar(f"Gana {local}", "Resultado", "goles", i > j)
    agregar("Empate", "Resultado", "goles", i == j)
    agregar(f"Gana {visita}", "Resultado", "goles", i < j)
    for equipo, cond in ((local, i > j), (visita, j > i)):
        for margen in (0.5, 1.5, 2.5):
            hcap = i - j if equipo == local else j - i
            agregar(f"{equipo} gana por más de {margen}", "Spread", "goles", hcap > margen)
    for linea in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        agregar(f"Más de {linea} goles", "Total goles", "goles", i + j > linea)
        agregar(f"Menos de {linea} goles", "Total goles", "goles", i + j < linea)
    agregar("Ambos equipos marcan: Sí", "Ambos marcan", "goles", (i > 0) & (j > 0))
    agregar("Ambos equipos marcan: No", "Ambos marcan", "goles", (i == 0) | (j == 0))
    for equipo, cond in ((local, i), (visita, j)):
        for linea in (0.5, 1.5, 2.5, 3.5):
            agregar(f"{equipo} más de {linea} goles", f"Total equipo {equipo}", "goles", cond > linea)

    # Primer gol estimado por tasas de anotación independientes.
    mu_l = float((p * i).sum())
    mu_v = float((p * j).sum())
    prob_sin_gol = float(p[0, 0])
    primero_l = (mu_l / max(mu_l + mu_v, 1e-9)) * (1 - prob_sin_gol)
    primero_v = (mu_v / max(mu_l + mu_v, 1e-9)) * (1 - prob_sin_gol)
    agregar(f"Primer gol: {local}", "Primer gol", "primer gol", None, primero_l)
    agregar(f"Primer gol: {visita}", "Primer gol", "primer gol", None, primero_v)
    agregar("Sin goles", "Primer gol", "primer gol", None, prob_sin_gol)

    if p_corners is not None and ci is not None and cj is not None:
        for equipo, cond in ((local, ci), (visita, cj)):
            for linea in range(1, 13):
                agregar(f"Corners {equipo}: {linea}+", f"Corners {equipo}", "corners", cond >= linea)
        for linea in (7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5):
            agregar(f"Más de {linea} corners totales", "Corners totales", "corners", ci + cj > linea)

    # Marcador exacto: se muestran solo los cinco resultados con mayor probabilidad.
    marcadores = []
    for goles_l in range(6):
        for goles_v in range(6):
            cond = (i == goles_l) & (j == goles_v)
            marcadores.append((float(p[cond].sum()), goles_l, goles_v, cond))
    marcadores.sort(reverse=True, key=lambda x: x[0])
    for prob, goles_l, goles_v, cond in marcadores[:5]:
        agregar(f"Marcador exacto {goles_l}-{goles_v}", "Marcador exacto", "goles", cond, prob)

    if resultado is not None:
        goles_l, goles_v = resultado
        for fila in filas:
            if fila["evento"] == "primer gol":
                if goles_l == 0 and goles_v == 0:
                    fila["acierto"] = fila["mercado"] == "Sin goles"
                elif goles_l > 0 and goles_v == 0:
                    fila["acierto"] = fila["mercado"] == f"Primer gol: {local}"
                elif goles_v > 0 and goles_l == 0:
                    fila["acierto"] = fila["mercado"] == f"Primer gol: {visita}"
                else:
                    fila["acierto"] = None
    return filas


def probabilidad_combo(legs, p_goles, p_corners):
    eventos = {}
    for leg in legs:
        eventos.setdefault(leg["evento"], []).append(leg)
    prob = 1.0
    for evento, sub in eventos.items():
        if evento == "primer gol" or any(x["condicion"] is None for x in sub):
            prob *= float(np.prod([x["p"] for x in sub]))
            continue
        matriz = p_goles if evento == "goles" else p_corners
        if matriz is None:
            return 0.0
        conjunta = np.logical_and.reduce([x["condicion"] for x in sub])
        prob *= float(matriz[conjunta].sum())
    return prob


def resumir_combinadas(mercados, p_goles, p_corners):
    candidatos = [m for m in mercados if 0.55 <= m["p"] <= 0.92]
    opciones = []
    for legs in combinations(candidatos, 2):
        if legs[0]["grupo"] == legs[1]["grupo"]:
            continue
        pc = probabilidad_combo(legs, p_goles, p_corners)
        if pc >= 0.25:
            resultado = None
            if all(x["acierto"] is not None for x in legs):
                resultado = all(x["acierto"] for x in legs)
            opciones.append((pc, legs, resultado))
    opciones.sort(key=lambda x: x[0], reverse=True)
    resumen, vistos = [], set()
    for pc, legs, resultado in opciones:
        llave = tuple(sorted(x["mercado"] for x in legs))
        if llave in vistos:
            continue
        vistos.add(llave)
        res = "Pendiente" if resultado is None else "✅ acertada" if resultado else "❌ fallada"
        resumen.append({"Combinada del partido": " + ".join(x["mercado"] for x in legs),
                        "Prob. estimada": f"{pc:.1%}", "Resultado": res})
        if len(resumen) == 5:
            break
    return resumen


try:
    clave_api = st.secrets["CLAVE"]
except Exception:
    st.error("Falta configurar el secreto CLAVE de football-data.org en la app.")
    st.stop()

hoy = datetime.now(TZ).date()
temporada = hoy.year if hoy.month >= 7 else hoy.year - 1
temporada_anterior = temporada - 1
codigos_corners = [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in (temporada_anterior, temporada)]

try:
    modelo = cargar_partidos_y_estadisticas(clave_api, temporada, temporada_anterior)
    partidos = modelo[0]
except requests.RequestException as exc:
    st.error(f"No pude cargar LaLiga desde football-data.org: {exc}")
    st.stop()
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"No pude preparar el modelo de LaLiga: {exc}")
    st.stop()

corners = cargar_corners(codigos_corners)
if corners is None:
    st.info("Los mercados de goles están disponibles. Football-data.co.uk todavía no publicó corners para LaLiga.")

inicio, fin = hoy, hoy + timedelta(days=2)
seleccionables = []
for partido in partidos:
    fecha_txt = partido.get("utcDate")
    if not fecha_txt:
        continue
    fecha_utc = datetime.fromisoformat(fecha_txt.replace("Z", "+00:00"))
    fecha_local = fecha_utc.astimezone(TZ).date()
    estado = partido.get("status")
    if inicio <= fecha_local <= fin and estado in {"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED", "FINISHED"}:
        seleccionables.append(partido)

if not seleccionables:
    st.info(f"No hay partidos de LaLiga entre el {inicio:%d/%m} y el {fin:%d/%m}.")
    st.stop()

seleccionables.sort(key=lambda m: m.get("utcDate", ""))
opciones = {}
for m in seleccionables:
    local = m["homeTeam"].get("shortName") or m["homeTeam"]["name"]
    visita = m["awayTeam"].get("shortName") or m["awayTeam"]["name"]
    fecha = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).astimezone(TZ)
    estado = " · FINAL" if m["status"] == "FINISHED" else " · EN JUEGO" if m["status"] in {"IN_PLAY", "PAUSED"} else ""
    marcador = m.get("score", {}).get("fullTime", {})
    final = f" · {marcador['home']}-{marcador['away']}" if m["status"] == "FINISHED" else ""
    etiqueta = f"{fecha:%a %d/%m %H:%M} · {local} vs {visita}{estado}{final}"
    opciones[etiqueta] = m

st.caption(f"Ventana: hoy {inicio:%d/%m} y próximos dos días hasta {fin:%d/%m} · hora local de Chicago")
pendientes = [m for m in seleccionables if m["status"] != "FINISHED"]
indice = 0
if pendientes:
    indice = next(i for i, m in enumerate(seleccionables) if m is pendientes[0])
etiqueta = st.selectbox("Partido", list(opciones), index=indice)
partido = opciones[etiqueta]
local = partido["homeTeam"].get("shortName") or partido["homeTeam"]["name"]
visita = partido["awayTeam"].get("shortName") or partido["awayTeam"]["name"]
gl, gv = estimar_goles(local, visita, modelo)
goles = np.arange(N_GOLES)
i, j = np.meshgrid(goles, goles, indexing="ij")
p_goles = np.outer(poisson.pmf(goles, gl), poisson.pmf(goles, gv))
p_goles /= p_goles.sum()

p_corners = ci = cj = None
corners_final = None
if corners is not None:
    lk = clave_equipo(local, corners["nombres"])
    vk = clave_equipo(visita, corners["nombres"])
    if lk and vk:
        lh = corners["equipos"][lk]
        va = corners["equipos"][vk]
        n_l = len(lh["home_for"])
        n_v = len(va["away_for"])
        media_l = (sum(lh["home_for"]) + K * corners["prom_home"]) / (n_l + K)
        contra_l = (sum(va["away_against"]) + K * corners["prom_home"]) / (n_v + K)
        media_v = (sum(va["away_for"]) + K * corners["prom_away"]) / (n_v + K)
        contra_v = (sum(lh["home_against"]) + K * corners["prom_away"]) / (n_l + K)
        p_corners, (ci, cj) = matriz_corner(((media_l + contra_l) / 2, (media_v + contra_v) / 2))
        corners_final = corners["resultados"].get((limpiar(local), limpiar(visita)))

resultado = None
if partido["status"] == "FINISHED":
    marcador = partido.get("score", {}).get("fullTime", {})
    if marcador.get("home") is not None and marcador.get("away") is not None:
        resultado = (int(marcador["home"]), int(marcador["away"]))

todos = calcular_mercados(p_goles, i, j, local, visita, resultado,
                          p_corners, ci, cj, corners_final)
combinadas = resumir_combinadas(todos, p_goles, p_corners)
st.subheader(f"{local} vs {visita}")
st.write(f"**Marcador esperado:** {gl:.1f}–{gv:.1f} goles")
if combinadas:
    st.markdown("### Resumen de combinadas para este partido")
    st.dataframe(pd.DataFrame(combinadas), use_container_width=True, hide_index=True)
else:
    st.info("No hay combinadas que superen los umbrales del modelo para este partido.")

selecciones = sorted([m for m in todos if m["p"] >= 0.55], key=lambda m: m["p"], reverse=True)[:8]
st.markdown("### Selecciones individuales destacadas")
if selecciones:
    st.dataframe(pd.DataFrame([
        {"Mercado": m["mercado"], "Probabilidad": f"{m['p']:.1%}",
         "Resultado": "✅ acertó" if m["acierto"] is True else "❌ falló" if m["acierto"] is False else "Pendiente"}
        for m in selecciones
    ]), use_container_width=True, hide_index=True)
else:
    st.caption("No hay una selección individual por encima del 55%.")

with st.expander("Ver todos los mercados modelados para este partido"):
    orden = ["Resultado", "Spread", "Total goles", "Ambos marcan", "Primer gol",
             "Corners totales", f"Corners {local}", f"Corners {visita}",
             f"Total equipo {local}", f"Total equipo {visita}", "Marcador exacto"]
    for grupo in orden:
        sub = [m for m in todos if m["grupo"] == grupo]
        if sub:
            st.markdown(f"**{grupo}**")
            st.dataframe(pd.DataFrame([
                {"Mercado": m["mercado"], "Probabilidad": f"{m['p']:.1%}",
                 "Resultado": "✅ acertó" if m["acierto"] is True else "❌ falló" if m["acierto"] is False else "Pendiente"}
                for m in sorted(sub, key=lambda x: x["p"], reverse=True)
            ]), use_container_width=True, hide_index=True)

st.caption(
    "Incluye mercados de partido: resultado 1X2, spread, total de goles, ambos marcan, "
    "marcador exacto, primer gol, goles por equipo y corners cuando hay datos. "
    "Los mercados de temporada (campeón, goleador, descenso) no son combinadas de partido."
)
st.warning(
    "Las probabilidades son estimaciones estadísticas, no cuotas ni recomendaciones de Kalshi. "
    "El resumen de combinada aproxima la relación entre mercados; confirma siempre el mercado "
    "y sus reglas en Kalshi antes de usarlo."
)
