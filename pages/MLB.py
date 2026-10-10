from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import nbinom


API = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 20

st.set_page_config(page_title="Predicciones MLB", page_icon="⚾", layout="wide")
st.title("⚾ MLB · predicciones y combinadas")
st.caption(
    "Forma de bateo reciente, carreras anotadas y permitidas, y pitchers abridores probables "
    "cuando MLB los publica. Las probabilidades reflejan incertidumbre de béisbol."
)

with st.expander("¿Qué mercados suelen aparecer en combinadas de MLB?", expanded=True):
    st.markdown(
        "**Home run de un bateador** es un mercado de props muy popular: FanDuel informó "
        "que fue su tipo de apuesta MLB con más volumen en 2025. Eso describe a esa casa, "
        "no a todas las casas ni un ranking de combinadas. También se ofrecen mercados de "
        "hits de bateadores, ponches de pitchers, ganador (moneyline), run line y carreras "
        "totales."
    )
    st.caption(
        "Esta página solo tiene datos para estimar ganador, run line y carreras. No inventa "
        "selecciones de home run, hits o ponches: faltan alineaciones, pitchers confirmados "
        "y líneas/cuotas. Las combinadas de abajo son cálculos del modelo, no las más "
        "apostadas por el público."
    )
    st.markdown(
        "Fuente: [FanDuel — Inside Baseball’s Hottest Betting Market]("
        "https://www.fanduel.com/about/news/going-yard-inside-baseball-hottest-betting-market-at-fanduel)"
    )


st.caption(
    "El historial MLB registra una sola captura inicial por contrato y lado. Tras la liquidación, "
    "WIN/LOSS se obtiene del resultado oficial de Kalshi; las predicciones históricas no se recalculan."
)

# Historial persistente: las filas se agregan en GitHub Actions y nunca se reescriben.
ROOT = Path(__file__).resolve().parent.parent
PRED_MLB = ROOT / "predicciones_mlb.csv"
RES_MLB = ROOT / "resultados_mlb.csv"
with st.expander("📚 Historial fijo MLB · predicciones, WIN y LOSS", expanded=True):
    try:
        if PRED_MLB.exists():
            hist = pd.read_csv(PRED_MLB, dtype=str).fillna("")
            if RES_MLB.exists():
                res = pd.read_csv(RES_MLB, dtype=str).fillna("")
                if not res.empty:
                    hist = hist.merge(res[["prediction_id", "estado", "resultado_kalshi", "marcador", "resuelto_en"]],
                                      on="prediction_id", how="left")
            if "estado" not in hist:
                hist["estado"] = ""
            hist["estado"] = hist["estado"].replace("", "PENDIENTE")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Capturas inmutables", len(hist))
            c2.metric("WIN", int((hist["estado"] == "WIN").sum()))
            c3.metric("LOSS", int((hist["estado"] == "LOSS").sum()))
            c4.metric("Pendientes", int((hist["estado"] == "PENDIENTE").sum()))
            filtro = st.selectbox("Filtrar historial", ["Todos", "WIN", "LOSS", "PENDIENTE"], key="mlb_hist_filtro")
            ver = hist if filtro == "Todos" else hist[hist["estado"] == filtro]
            cols = [x for x in ["fecha", "visita", "local", "categoria", "mercado", "lado",
                                "probabilidad_modelo", "precio_captura", "estado", "marcador",
                                "ticker", "capturado_en"] if x in ver.columns]
            st.dataframe(ver[cols].sort_values("capturado_en", ascending=False) if not ver.empty else ver[cols],
                         use_container_width=True, hide_index=True)
            st.download_button("Descargar historial MLB CSV", hist.to_csv(index=False).encode("utf-8"),
                               file_name="historial_mlb.csv", mime="text/csv")
        else:
            st.info("El historial se creará en la primera ejecución automática del registrador MLB.")
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"El historial MLB aún no se pudo leer: {exc}")


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
def cargar_partidos_recientes(inicio, fin):
    return cargar_calendario(inicio, fin)


def construir_forma(equipos_juegos, antes_de):
    forma = {}
    por_equipo = {}
    for juego in equipos_juegos:
        if juego.get("status", {}).get("abstractGameState") != "Final":
            continue
        fecha = juego.get("gameDate", "")
        if fecha >= antes_de:
            continue
        equipos = juego.get("teams", {})
        home = equipos.get("home", {})
        away = equipos.get("away", {})
        equipo_h = home.get("team", {}).get("id")
        equipo_a = away.get("team", {}).get("id")
        runs_h, runs_a = home.get("score"), away.get("score")
        if equipo_h is None or equipo_a is None or runs_h is None or runs_a is None:
            continue
        por_equipo.setdefault(int(equipo_h), []).append((fecha, float(runs_h), float(runs_a)))
        por_equipo.setdefault(int(equipo_a), []).append((fecha, float(runs_a), float(runs_h)))
    for equipo_id, juegos in por_equipo.items():
        recientes = sorted(juegos, reverse=True)[:10]
        if recientes:
            forma[equipo_id] = {
                "juegos": len(recientes),
                "anotadas": float(np.mean([x[1] for x in recientes])),
                "permitidas": float(np.mean([x[2] for x in recientes])),
            }
    return forma


@st.cache_data(ttl=1800, show_spinner=False)
def cargar_pitcher(pitcher_id, season):
    data = get_json(
        f"people/{pitcher_id}/stats",
        {"stats": "season", "group": "pitching", "season": season},
    )
    for bloque in data.get("stats", []):
        for split in bloque.get("splits", []):
            stat = split.get("stat", {})
            if stat.get("era") not in (None, "-"):
                try:
                    return {
                        "nombre": split.get("player", {}).get("fullName", "Pitcher probable"),
                        "era": float(stat["era"]),
                        "innings": float(stat.get("inningsPitched") or 0),
                    }
                except (TypeError, ValueError):
                    pass
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def cargar_era_liga(season):
    data = get_json(
        "teams/stats",
        {"stats": "season", "group": "pitching", "season": season, "sportIds": 1},
    )
    eras = []
    for bloque in data.get("stats", []):
        for split in bloque.get("splits", []):
            try:
                era = float(split.get("stat", {}).get("era"))
                if np.isfinite(era) and era > 0:
                    eras.append(era)
            except (TypeError, ValueError):
                continue
    return float(np.mean(eras)) if eras else 4.20


@st.cache_data(ttl=900, show_spinner=False)
def cargar_calendario(inicio, fin):
    data = get_json(
        "schedule",
        {
            "sportId": 1,
            "startDate": inicio,
            "endDate": fin,
            "gameTypes": "R,F,D,L,W",
            "hydrate": "probablePitcher",
        },
    )
    partidos = []
    for jornada in data.get("dates", []):
        partidos.extend(jornada.get("games", []))
    return partidos


def estimar_carreras(local_id, visita_id, stats, forma, promedio_liga, era_liga,
                     abridor_local=None, abridor_visita=None):
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

    # La forma de los últimos diez partidos cuenta, pero se mezcla con el año completo
    # para que una racha corta no domine el pronóstico.
    forma_l, forma_v = forma.get(local_id), forma.get(visita_id)
    if forma_l:
        peso = min(forma_l["juegos"] / 10, 0.40)
        r_l = (1 - peso) * r_l + peso * (0.65 * forma_l["anotadas"] + 0.35 * promedio_liga)
    if forma_v:
        peso = min(forma_v["juegos"] / 10, 0.40)
        r_v = (1 - peso) * r_v + peso * (0.65 * forma_v["anotadas"] + 0.35 * promedio_liga)

    # Ajusta el ataque por el rendimiento reciente permitido por el rival.
    if forma_v:
        peso = min(forma_v["juegos"] / 10, 0.25)
        r_l = (1 - peso) * r_l + peso * (0.65 * forma_v["permitidas"] + 0.35 * promedio_liga)
    if forma_l:
        peso = min(forma_l["juegos"] / 10, 0.25)
        r_v = (1 - peso) * r_v + peso * (0.65 * forma_l["permitidas"] + 0.35 * promedio_liga)

    # El ERA del abridor se contrae hacia la media; muestras pequeñas tienen poco peso.
    liga_era = era_liga or 4.20
    if abridor_visita:
        peso = min(abridor_visita["innings"] / 45, 0.70)
        era = peso * abridor_visita["era"] + (1 - peso) * liga_era
        r_l *= float(np.clip(era / liga_era, 0.82, 1.18))
    if abridor_local:
        peso = min(abridor_local["innings"] / 45, 0.70)
        era = peso * abridor_local["era"] + (1 - peso) * liga_era
        r_v *= float(np.clip(era / liga_era, 0.82, 1.18))

    # Ventaja local pequeña; no da por hecho que el local ganará.
    return max(1.8, r_l * 1.03), max(1.8, r_v * 0.98)


def distribucion_carreras(media, eje):
    # Sobredispersión moderada: los marcadores de béisbol varían más que un Poisson simple.
    dispersion = 0.12
    n = 1 / dispersion
    p = n / (n + media)
    return nbinom.pmf(eje, n, p)


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
        for linea in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5):
            agregar(f"{nombre} más de {linea} carreras", f"Total {nombre}", expr > linea)
            agregar(f"{nombre} menos de {linea} carreras", f"Total {nombre}", expr < linea)
        for exactas in range(0, 16):
            agregar(f"{nombre}: exactamente {exactas} carreras", f"Exacto {nombre}", expr == exactas)

    for exactas in range(0, 21):
        agregar(f"Total exacto: {exactas} carreras", "Total exacto", total == exactas)
    margen_local = puntos[0] - puntos[1]
    margen_visita = puntos[1] - puntos[0]
    for margen in (1, 2, 3, 4):
        agregar(f"{local} gana por exactamente {margen}", "Margen", margen_local == margen)
        agregar(f"{visita} gana por exactamente {margen}", "Margen", margen_visita == margen)
    agregar(f"{local} gana por 5 o más", "Margen", margen_local >= 5)
    agregar(f"{visita} gana por 5 o más", "Margen", margen_visita >= 5)

    candidatos = []
    for ih in range(prob.shape[0]):
        for ia in range(prob.shape[1]):
            candidatos.append((float(prob[ih, ia]), ih, ia))
    for _, ih, ia in sorted(candidatos, reverse=True)[:15]:
        agregar(f"Marcador exacto {local} {ih}-{ia} {visita}", "Marcador exacto",
                (puntos[0] == ih) & (puntos[1] == ia))
    return filas


def recomendar_combos(probabilidades, filas, max_combos=5):
    # Prioriza estructuras habituales de SGP y conserva la dependencia del marcador.
    pares_habituales = {
        frozenset(("Ganador", "Total")),
        frozenset(("Ganador", "Run line")),
        frozenset(("Ganador", "Total {equipo}")),
        frozenset(("Run line", "Total")),
        frozenset(("Run line", "Total {equipo}")),
        frozenset(("Total", "Total {equipo}")),
    }

    def tipo(grupo):
        return "Total {equipo}" if grupo.startswith("Total ") else grupo

    candidatos = [
        f for f in filas
        if 0.55 <= f["p"] <= 0.90
        and not f["mercado"].endswith("más de 0.5 carreras")
    ]
    combos = []
    for legs in combinations(candidatos, 2):
        if legs[0]["grupo"] == legs[1]["grupo"]:
            continue
        if frozenset(tipo(x["grupo"]) for x in legs) not in pares_habituales:
            continue
        conjunta = legs[0]["condicion"] & legs[1]["condicion"]
        p_combo = float(probabilidades[conjunta].sum())
        if 0.25 <= p_combo <= 0.78:
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
inicio_forma = hoy - timedelta(days=30)

try:
    stats = cargar_stats(temporada)
    try:
        era_liga = cargar_era_liga(temporada)
    except (requests.RequestException, ValueError, KeyError, TypeError):
        era_liga = 4.20
    partidos = cargar_calendario(inicio.isoformat(), fin.isoformat())
    juegos_forma = cargar_partidos_recientes(inicio_forma.isoformat(), (hoy - timedelta(days=1)).isoformat())
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
    f"promedio {promedio:.2f} carreras/equipo · últimos 10 juegos disponibles · "
    "datos actualizados cada 15 minutos."
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
    abridor_local = juego.get("teams", {}).get("home", {}).get("probablePitcher")
    abridor_visita = juego.get("teams", {}).get("away", {}).get("probablePitcher")
    datos_abridor_local = datos_abridor_visita = None
    for probable, destino in ((abridor_local, "local"), (abridor_visita, "visita")):
        if probable and probable.get("id"):
            try:
                datos = cargar_pitcher(int(probable["id"]), temporada)
            except (requests.RequestException, ValueError, KeyError, TypeError):
                datos = None
            if destino == "local":
                datos_abridor_local = datos
            else:
                datos_abridor_visita = datos
    fecha_juego = juego.get("gameDate", "9999")
    forma_juego = construir_forma(juegos_forma, fecha_juego)
    mu_h, mu_a = estimar_carreras(
        int(home["id"]), int(away["id"]), stats, forma_juego, promedio, era_liga,
        datos_abridor_local, datos_abridor_visita,
    )
    matriz_prob = np.outer(distribucion_carreras(eje, mu_h), distribucion_carreras(eje, mu_a))
    matriz_prob /= matriz_prob.sum()
    fecha = juego.get("gameDate", "")[:16].replace("T", " ")
    marcador = f" · Final {a_runs}-{h_runs}" if status == "Final" else ""
    with st.expander(f"{fecha} · {game['away']} @ {game['home']}{marcador}", expanded=status == "Preview"):
        st.write(f"**Marcador esperado:** {mu_a:.1f}–{mu_h:.1f} carreras (visita–local)")
        forma_l = forma_juego.get(int(home["id"]))
        forma_v = forma_juego.get(int(away["id"]))
        if forma_l or forma_v:
            st.caption(
                "Últimos juegos: "
                f"{game['home']} anotó/permitió {forma_l['anotadas']:.1f}/{forma_l['permitidas']:.1f} por juego · "
                f"{game['away']} {forma_v['anotadas']:.1f}/{forma_v['permitidas']:.1f} por juego"
                if forma_l and forma_v else
                "Forma reciente disponible solo para uno de los dos equipos; el otro usa promedio de temporada."
            )
        estado_abridores = []
        for etiqueta_abridor, raw, calculado in (
            ("Local", abridor_local, datos_abridor_local),
            ("Visita", abridor_visita, datos_abridor_visita),
        ):
            nombre = (calculado or {}).get("nombre") or (raw or {}).get("fullName")
            if nombre:
                era_txt = f" · ERA {calculado['era']:.2f}" if calculado else " · estadísticas no disponibles"
                estado_abridores.append(f"{etiqueta_abridor}: {nombre}{era_txt}")
            else:
                estado_abridores.append(f"{etiqueta_abridor}: sin abridor probable publicado")
        st.caption("Abridores · " + " | ".join(estado_abridores))
        picks = mercados(matriz_prob, (matriz_local, matriz_visita), game)
        picks_ordenados = sorted(picks, key=lambda x: x["p"], reverse=True)
        recomendados = [
            x for x in picks_ordenados
            if 0.55 <= x["p"] <= 0.88
            and not x["mercado"].endswith("más de 0.5 carreras")
        ][:5]
        if recomendados:
            tabla = pd.DataFrame([
                {"Mercado": x["mercado"], "Prob. estimada": f"{x['p']:.1%}",
                 "Resultado": "✅ acertó" if x["acierto"] is True else
                 "❌ falló" if x["acierto"] is False else "Pendiente"}
                for x in recomendados
            ])
            st.markdown("**Resumen: líneas razonables por probabilidad estimada**")
            st.dataframe(tabla, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No hay selecciones en el rango 55–88% después de excluir líneas casi "
                "automáticas. Revisa todos los mercados si quieres ver el resto."
            )
        combo_rows = recomendar_combos(matriz_prob, picks)
        if combo_rows:
            st.markdown("**Combinadas de mercados habituales (2 selecciones)**")
            st.caption(
                "Ordenadas por probabilidad conjunta estimada. Una probabilidad alta no "
                "indica cuota rentable; comprueba que ambas selecciones y sus líneas estén "
                "disponibles en tu casa."
            )
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
    f"El modelo mezcla producción de temporada y últimos 10 resultados, incorpora el ERA "
    f"medio de liga ({era_liga:.2f}) y el ERA del "
    "abridor probable cuando está publicado y usa una distribución con mayor variabilidad "
    "que Poisson. Aún no incorpora alineaciones confirmadas, lesiones, parque/clima, cuotas "
    "ni props individuales; sus porcentajes son estimaciones, no garantías."
)
