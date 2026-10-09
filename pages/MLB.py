from datetime import date, datetime, timedelta
from itertools import combinations
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson


API = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 20

st.set_page_config(page_title="Predicciones MLB", page_icon="⚾", layout="wide")
st.title("⚾ MLB · predicciones y combinadas")
st.caption(
    "Página independiente de fútbol. Usa calendario y resultados de MLB y estadísticas "
    "de carreras de la temporada para estimar moneyline, run line y totales."
)


def get_json(path, params):
    response = requests.get(f"{API}/{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=1800, show_spinner=False)
def cargar_stats(season):
    # El endpoint oficial devuelve estadísticas de bateo por equipo.
    data = get_json(
        "teams/stats",
        {"stats": "season", "group": "hitting", "season": season, "sportIds": 1},
    )
    equipos = {}
    for bloque in data.get("stats", []):
        for split in bloque.get("splits", []):
            equipo = split.get("team", {})
            stat = split.get("stat", {})
            juegos = float(stat.get("gamesPlayed") or 0)
            carreras = float(stat.get("runs") or 0)
            if equipo.get("id") and juegos > 0:
                equipos[int(equipo["id"])] = {
                    "nombre": equipo.get("name", "Equipo"),
                    "rpg": carreras / juegos,
                    "juegos": juegos,
                }
    if not equipos:
        raise ValueError("MLB no devolvió estadísticas de carreras para esta temporada.")
    return equipos


@st.cache_data(ttl=900, show_spinner=False)
def cargar_calendario(inicio, fin):
    data = get_json(
        "schedule",
        {
            "sportId": 1,
            "startDate": inicio,
            "endDate": fin,
            "gameTypes": "R,F,D,L,W",
        },
    )
    partidos = []
    for jornada in data.get("dates", []):
        partidos.extend(jornada.get("games", []))
    return partidos


def estimar_carreras(local_id, visita_id, stats, promedio_liga):
    local = stats.get(local_id)
    visita = stats.get(visita_id)
    if local and visita:
        # Regresión hacia la media para que equipos con pocos juegos no dominen el cálculo.
        peso_l = min(local["juegos"] / 40, 1)
        peso_v = min(visita["juegos"] / 40, 1)
        r_l = peso_l * local["rpg"] + (1 - peso_l) * promedio_liga
        r_v = peso_v * visita["rpg"] + (1 - peso_v) * promedio_liga
    else:
        r_l = r_v = promedio_liga
    return max(1.8, r_l * 1.04), max(1.8, r_v / 1.04)


def mercados(prob, puntos, partido):
    local, visita = partido["home"], partido["away"]
    final = partido.get("status") == "Final"
    carreras_l = partido.get("runs_home")
    carreras_v = partido.get("runs_away")
    filas = []

    def agregar(nombre, grupo, condicion):
        p = float(prob[condicion].sum())
        acierto = None
        if final and carreras_l is not None and carreras_v is not None:
            acierto = bool(condicion[int(carreras_l), int(carreras_v)])
        filas.append({"mercado": nombre, "grupo": grupo, "p": p, "acierto": acierto,
                      "condicion": condicion})

    agregar(f"Gana {local} (moneyline)", "Ganador", puntos[0] > puntos[1])
    agregar(f"Gana {visita} (moneyline)", "Ganador", puntos[1] > puntos[0])
    agregar(f"{local} +1.5 carreras", "Run line", puntos[0] + 1.5 > puntos[1])
    agregar(f"{visita} +1.5 carreras", "Run line", puntos[1] + 1.5 > puntos[0])
    total = puntos[0] + puntos[1]
    for linea in (6.5, 7.5, 8.5, 9.5, 10.5, 11.5):
        agregar(f"Más de {linea} carreras", "Total", total > linea)
        agregar(f"Menos de {linea} carreras", "Total", total < linea)
    for nombre, expr in ((local, puntos[0]), (visita, puntos[1])):
        for linea in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
            agregar(f"{nombre} más de {linea} carreras", f"Total {nombre}", expr > linea)
    return filas


def recomendar_combos(probabilidades, filas, max_combos=5):
    # Producto de la distribución conjunta del marcador: conserva la dependencia entre carreras.
    candidatos = [f for f in filas if f["p"] >= 0.55]
    combos = []
    for legs in combinations(candidatos, 2):
        if legs[0]["grupo"] == legs[1]["grupo"]:
            continue
        conjunta = legs[0]["condicion"] & legs[1]["condicion"]
        p_combo = float(probabilidades[conjunta].sum())
        if p_combo >= 0.25:
            combos.append((p_combo, legs))
    combos.sort(key=lambda x: x[0], reverse=True)
    salida, vistos = [], set()
    for p_combo, legs in combos:
        llave = tuple(sorted(x["mercado"] for x in legs))
        if llave in vistos:
            continue
        vistos.add(llave)
        acertada = all(x["acierto"] is True for x in legs) if all(x["acierto"] is not None for x in legs) else None
        salida.append({"Combinada": " + ".join(x["mercado"] for x in legs),
                       "Prob. estimada": f"{p_combo:.1%}", "Resultado":
                       "✅ acertada" if acertada else "❌ fallada" if acertada is False else "Pendiente"})
        if len(salida) == max_combos:
            break
    return salida


hoy = datetime.now(ZoneInfo("America/Chicago")).date()
temporada = hoy.year
inicio = hoy
fin = hoy + timedelta(days=2)

try:
    stats = cargar_stats(temporada)
    partidos = cargar_calendario(inicio.isoformat(), fin.isoformat())
except requests.RequestException as exc:
    st.error(f"No pude conectar con los datos de MLB: {exc}")
    st.stop()
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"No pude preparar las predicciones MLB: {exc}")
    st.stop()

if not partidos:
    st.info("No hay partidos hoy ni en los próximos dos días.")
    st.stop()

promedio = float(np.mean([x["rpg"] for x in stats.values()]))
eje = np.arange(0, 21)
matriz_local, matriz_visita = np.meshgrid(eje, eje, indexing="ij")
st.caption(
    f"Partidos del {inicio:%d/%m} al {fin:%d/%m} · temporada {temporada} · "
    f"promedio de liga {promedio:.2f} carreras/equipo · datos actualizados cada 15 minutos."
)

partidos = sorted(partidos, key=lambda p: p.get("gameDate", ""))
opciones = {}
for juego in partidos:
    local = juego.get("teams", {}).get("home", {}).get("team", {}).get("name", "Local")
    visita = juego.get("teams", {}).get("away", {}).get("team", {}).get("name", "Visita")
    hora = juego.get("gameDate", "")[:16].replace("T", " ")
    estado = juego.get("status", {}).get("abstractGameState", "")
    marcador = " · FINAL" if estado == "Final" else " · EN JUEGO" if estado == "Live" else ""
    clave = f"{hora} · {visita} @ {local}{marcador}"
    opciones[clave] = juego

indice_inicial = next(
    (i for i, partido in enumerate(partidos)
     if partido.get("status", {}).get("abstractGameState") != "Final"),
    0,
)
seleccion = st.selectbox("Elige un partido (solo se muestran sus predicciones)", list(opciones), index=indice_inicial)
juego = opciones[seleccion]
for juego in [juego]:
    home = juego.get("teams", {}).get("home", {}).get("team", {})
    away = juego.get("teams", {}).get("away", {}).get("team", {})
    if not home.get("id") or not away.get("id"):
        continue
    status = juego.get("status", {}).get("abstractGameState", "Preview")
    h_runs = juego.get("teams", {}).get("home", {}).get("score")
    a_runs = juego.get("teams", {}).get("away", {}).get("score")
    game = {
        "home": home.get("name", "Local"), "away": away.get("name", "Visita"),
        "status": status,
        "runs_home": h_runs if status == "Final" else None,
        "runs_away": a_runs if status == "Final" else None,
    }
    mu_h, mu_a = estimar_carreras(int(home["id"]), int(away["id"]), stats, promedio)
    matriz_prob = np.outer(poisson.pmf(eje, mu_h), poisson.pmf(eje, mu_a))
    matriz_prob /= matriz_prob.sum()
    fecha = juego.get("gameDate", "")[:16].replace("T", " ")
    marcador = f" · Final {a_runs}-{h_runs}" if status == "Final" else ""
    with st.expander(f"{fecha} · {game['away']} @ {game['home']}{marcador}", expanded=status == "Preview"):
        st.write(f"**Marcador esperado:** {mu_a:.1f}–{mu_h:.1f} carreras (visita–local)")
        picks = mercados(matriz_prob, (matriz_local, matriz_visita), game)
        picks_ordenados = sorted(picks, key=lambda x: x["p"], reverse=True)
        recomendados = [x for x in picks_ordenados if x["p"] >= 0.55][:5]
        if recomendados:
            tabla = pd.DataFrame([
                {"Mercado": x["mercado"], "Prob. estimada": f"{x['p']:.1%}",
                 "Resultado": "✅ acertó" if x["acierto"] is True else
                 "❌ falló" if x["acierto"] is False else "Pendiente"}
                for x in recomendados
            ])
            st.markdown("**Selecciones destacadas**")
            st.dataframe(tabla, use_container_width=True, hide_index=True)
        else:
            st.info("El modelo no encuentra selecciones por encima del 55% para este partido.")
        combo_rows = recomendar_combos(matriz_prob, picks)
        if combo_rows:
            st.markdown("**Combinadas sugeridas (2 selecciones)**")
            st.dataframe(pd.DataFrame(combo_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No hay combinadas con el umbral mínimo de probabilidad configurado.")
        with st.expander("Ver todos los mercados calculados"):
            st.dataframe(pd.DataFrame([
                {"Mercado": x["mercado"], "Probabilidad": f"{x['p']:.1%}",
                 "Resultado": "✅ acertó" if x["acierto"] is True else
                 "❌ falló" if x["acierto"] is False else "Pendiente"}
                for x in picks_ordenados
            ]), use_container_width=True, hide_index=True)

st.warning(
    "Modelo inicial: solo usa carreras por equipo de la temporada. No incorpora todavía "
    "pitcher abridor, lesiones, cuotas ni props individuales (hits, ponches o home runs); "
    "por eso estas probabilidades no son recomendaciones financieras ni garantía de acierto."
)
