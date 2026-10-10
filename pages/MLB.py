from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import nbinom, poisson


API = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 20

st.set_page_config(page_title="Predicciones MLB", page_icon="⚾", layout="wide")
st.title("⚾ MLB · predicciones por partido")
st.caption(
    "Solo se muestran unas pocas selecciones por partido cuando la proyección es suficientemente favorable. "
    "Las opciones con probabilidad baja o sin datos sólidos se omiten."
)

with st.expander("Mercados MLB que se pueden proyectar", expanded=False):
    st.markdown(
        "- **Partido/equipos:** ganador, run line, total de carreras y total por equipo. "
        "Marcador exacto y márgenes se dejan en el detalle porque son menos estables.\n"
        "- **Pitchers:** ponches y, cuando hay muestra suficiente, hits permitidos, carreras limpias y bases por bolas.\n"
        "- **Bateadores:** 1+ hit, bases totales, carreras impulsadas y home run.\n"
        "- **Entradas y mercados especiales:** se muestran en el historial de Kalshi del partido si están disponibles, "
        "pero no reciben una probabilidad propia sin datos por entrada suficientes."
    )


st.caption("Las selecciones y su historial se muestran por partido. Las capturas originales se conservan y WIN/LOSS se añade tras la liquidación oficial.")

# Leer historial en segundo plano; la interfaz lo filtra por el partido seleccionado.
ROOT = Path(__file__).resolve().parent.parent
PRED_MLB = ROOT / "predicciones_mlb.csv"
RES_MLB = ROOT / "resultados_mlb.csv"

def leer_historial_mlb():
    try:
        pred = pd.read_csv(PRED_MLB, dtype=str).fillna("") if PRED_MLB.exists() else pd.DataFrame()
        res = pd.read_csv(RES_MLB, dtype=str).fillna("") if RES_MLB.exists() else pd.DataFrame()
        if not pred.empty and not res.empty and "prediction_id" in pred and "prediction_id" in res:
            cols_res = [x for x in ["prediction_id", "estado", "resultado_kalshi", "marcador", "resuelto_en"] if x in res]
            pred = pred.merge(res[cols_res], on="prediction_id", how="left")
        return pred
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return pd.DataFrame()

historial_mlb = leer_historial_mlb()

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
                        "strikeouts": float(stat.get("strikeOuts") or 0),
                        "hits_allowed": float(stat.get("hits") or 0),
                        "walks": float(stat.get("baseOnBalls") or 0),
                        "earned_runs": float(stat.get("earnedRuns") or 0),
                    }
                except (TypeError, ValueError):
                    pass
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def cargar_bateadores_equipo(team_id, season):
    """Season batting stats for candidates; does not imply a confirmed lineup."""
    data = get_json("stats", {
        "stats": "season", "group": "hitting", "season": season,
        "sportIds": 1, "teamId": int(team_id), "limit": 1000,
    })
    rows = []
    for block in data.get("stats", []):
        for split in block.get("splits", []):
            player, stat = split.get("player", {}), split.get("stat", {})
            try:
                pa = float(stat.get("plateAppearances") or 0)
                ab = float(stat.get("atBats") or 0)
                hits = float(stat.get("hits") or 0)
                hr = float(stat.get("homeRuns") or 0)
                total_bases = float(stat.get("totalBases") or 0)
                rbi = float(stat.get("rbi") or 0)
                stolen_bases = float(stat.get("stolenBases") or 0)
                avg = float(stat.get("avg") or (hits / ab if ab else 0))
                if player.get("id") and pa >= 30:
                    rows.append({"nombre": player.get("fullName", "Bateador"), "pa": pa,
                                 "ab": ab, "hits": hits, "hr": hr, "avg": avg,
                                 "total_bases": total_bases, "rbi": rbi,
                                 "stolen_bases": stolen_bases})
            except (TypeError, ValueError):
                continue
    unique = {row["nombre"]: row for row in rows}
    return sorted(unique.values(), key=lambda x: (x["pa"], x["hr"]), reverse=True)


def crear_apuestas_sencillas(juego, datos_abridor_local, datos_abridor_visita, temporada):
    picks = []
    for lado, equipo in (("Local", "home"), ("Visita", "away")):
        raw = juego.get("teams", {}).get(equipo, {}).get("probablePitcher") or {}
        pitcher = datos_abridor_local if lado == "Local" else datos_abridor_visita
        if raw.get("id") and pitcher and pitcher.get("innings", 0) > 0:
            k_por_9 = pitcher.get("strikeouts", 0) / max(pitcher["innings"], 1.0) * 9
            k_media = min(7.5, max(1.5, k_por_9 * 5.0 / 9))
            p_over = float(poisson.sf(4, k_media))
            picks.append({
                "Tipo": "Pitcher · ponches",
                "Jugada sencilla": f'{pitcher["nombre"]} más de 4.5 ponches',
                "Probabilidad estimada": p_over,
                "Dato base": f'{pitcher.get("strikeouts", 0):.0f} K en {pitcher["innings"]:.1f} entradas',
                "Nota": "Tasa de ponches con 5 entradas esperadas",
            })
            picks.append({
                "Tipo": "Pitcher · ponches",
                "Jugada sencilla": f'{pitcher["nombre"]} menos de 4.5 ponches',
                "Probabilidad estimada": 1 - p_over,
                "Dato base": f'{pitcher.get("strikeouts", 0):.0f} K en {pitcher["innings"]:.1f} entradas',
                "Nota": "Complemento del over; depende de las entradas lanzadas",
            })
            if pitcher["innings"] >= 15:
                for key, line, label, desc in (
                    ("hits_allowed", 4.5, "hits permitidos", "Hits permitidos por 5 entradas esperadas"),
                    ("earned_runs", 2.5, "carreras limpias", "Carreras limpias por 5 entradas esperadas"),
                    ("walks", 1.5, "bases por bolas", "Bases por bolas por 5 entradas esperadas"),
                ):
                    media = max(0.05, pitcher.get(key, 0) / pitcher["innings"] * 5)
                    p_over = float(poisson.sf(int(line), media))
                    picks.append({
                        "Tipo": "Pitcher · " + label,
                        "Jugada sencilla": f'{pitcher["nombre"]} más de {line} {label}',
                        "Probabilidad estimada": p_over,
                        "Dato base": f'{pitcher.get(key, 0):.0f} en {pitcher["innings"]:.1f} entradas',
                        "Nota": desc + "; proyección aproximada, no línea confirmada",
                    })

    for lado, equipo_key in (("Local", "home"), ("Visita", "away")):
        team = juego.get("teams", {}).get(equipo_key, {}).get("team", {})
        if not team.get("id"):
            continue
        try:
            bateadores = cargar_bateadores_equipo(int(team["id"]), temporada)
        except (requests.RequestException, ValueError, KeyError, TypeError):
            bateadores = []
        elegibles = [b for b in bateadores if b["pa"] >= 80 and b["ab"] > 0]
        elegibles.sort(key=lambda b: (b["avg"], b["pa"]), reverse=True)
        for b in elegibles[:2]:
            p_hit = 1 - (1 - min(0.75, max(0.01, b["avg"]))) ** 4
            picks.append({
                "Tipo": "Bateador · hits",
                "Jugada sencilla": f'{b["nombre"]} 1+ hit',
                "Probabilidad estimada": min(0.95, max(0.05, p_hit)),
                "Dato base": f'{b["hits"]:.0f} hits / {b["ab"]:.0f} turnos · AVG {b["avg"]:.3f}',
                "Nota": f'Candidato de {lado.lower()}; confirmar alineación titular',
            })
        hr_candidates = [b for b in bateadores if b["pa"] >= 80]
        hr_candidates.sort(key=lambda b: b["hr"] / max(b["pa"], 1), reverse=True)
        for b in hr_candidates[:1]:
            tasa = min(0.20, max(0.001, b["hr"] / max(b["pa"], 1)))
            picks.append({
                "Tipo": "Bateador · home run",
                "Jugada sencilla": f'{b["nombre"]} conecta HR',
                "Probabilidad estimada": min(0.65, max(0.01, 1 - (1 - tasa) ** 4)),
                "Dato base": f'{b["hr"]:.0f} HR en {b["pa"]:.0f} apariciones al plato',
                "Nota": f'Candidato de {lado.lower()}; confirmar alineación titular',
            })
        for b in [x for x in bateadores if x["pa"] >= 120 and x["total_bases"] > 0][:3]:
            media_tb = max(0.05, b["total_bases"] / b["pa"] * 4)
            p_tb = float(poisson.sf(1, media_tb))
            picks.append({
                "Tipo": "Bateador · bases totales",
                "Jugada sencilla": f'{b["nombre"]} más de 1.5 bases totales',
                "Probabilidad estimada": min(0.90, max(0.05, p_tb)),
                "Dato base": f'{b["total_bases"]:.0f} bases totales en {b["pa"]:.0f} apariciones',
                "Nota": f'Candidato de {lado.lower()}; confirmar alineación titular',
            })
        rbi_candidates = [x for x in bateadores if x["pa"] >= 120 and x["rbi"] > 0]
        rbi_candidates.sort(key=lambda x: x["rbi"] / x["pa"], reverse=True)
        for b in rbi_candidates[:1]:
            p_rbi = 1 - (1 - min(0.35, b["rbi"] / b["pa"])) ** 4
            picks.append({
                "Tipo": "Bateador · carreras impulsadas",
                "Jugada sencilla": f'{b["nombre"]} 1+ carrera impulsada',
                "Probabilidad estimada": min(0.85, max(0.05, p_rbi)),
                "Dato base": f'{b["rbi"]:.0f} RBI en {b["pa"]:.0f} apariciones',
                "Nota": f'Candidato de {lado.lower()}; depende de la posición en el lineup',
            })
    return sorted(picks, key=lambda x: x["Probabilidad estimada"], reverse=True)


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

        props_sencillas = crear_apuestas_sencillas(
            juego, datos_abridor_local, datos_abridor_visita, temporada
        )
        picks = mercados(matriz_prob, (matriz_local, matriz_visita), game)
        picks_ordenados = sorted(picks, key=lambda x: x["p"], reverse=True)

        # Selecciones sencillas del partido: solo categorías modelables y sin mezclar otros juegos.
        # Umbral conservador: mostrar solo probabilidades >=62%, y solo mercados principales.
        # Una selección por categoría evita llenar la página con líneas parecidas.
        candidatas = [
            x for x in picks_ordenados
            if 0.62 <= x["p"] <= 0.85
            and x["grupo"] in {"Ganador", "Run line", "Total"}
        ]
        por_categoria = {}
        for x in candidatas:
            if x["grupo"] not in por_categoria or x["p"] > por_categoria[x["grupo"]]["p"]:
                por_categoria[x["grupo"]] = x
        recomendados = sorted(por_categoria.values(), key=lambda x: x["p"], reverse=True)[:3]
        if recomendados:
            st.markdown("### Selecciones sencillas recomendadas para este partido")
            for x in recomendados:
                with st.container(border=True):
                    st.markdown(f"**{x['mercado']}**")
                    st.write(f"Probabilidad estimada: **{x['p']:.1%}**")
                    if x["acierto"] is not None:
                        st.write("Resultado del modelo: " + ("✅ acertó" if x["acierto"] else "❌ falló"))
                    st.caption("Proyección estadística del partido; confirma la línea y la cuota antes de apostar.")
        else:
            st.info("Este partido no tiene una selección principal que supere el filtro conservador de probabilidad (62%). Mejor no forzar una apuesta.")

        props_filtradas = [x for x in props_sencillas if 0.65 <= x["Probabilidad estimada"] <= 0.85]
        # Una prop por tipo y como máximo tres: evitar picks marginales y duplicados.
        props_por_tipo = {}
        for x in sorted(props_filtradas, key=lambda y: y["Probabilidad estimada"], reverse=True):
            props_por_tipo.setdefault(x["Tipo"], x)
        props_recomendadas = list(props_por_tipo.values())[:3]
        if props_recomendadas:
            st.markdown("### Props individuales con mejor proyección")
            st.caption("Solo props con probabilidad estimada entre 65% y 85%; los candidatos de bateo siguen sujetos a confirmar la alineación.")
            for x in props_recomendadas:
                with st.container(border=True):
                    st.markdown(f"**{x['Jugada sencilla']}**")
                    st.caption(x["Tipo"])
                    st.write(f"Probabilidad estimada: **{x['Probabilidad estimada']:.1%}**")
                    st.caption(f"{x['Dato base']} · {x['Nota']}")
        else:
            st.info("No hay props individuales que superen el filtro de probabilidad y datos. Se omiten las opciones más inciertas.")

        # Mostrar únicamente el historial/los contratos que corresponden al partido elegido.
        game_id = str(juego.get("gamePk", ""))
        if not historial_mlb.empty and "match_id" in historial_mlb.columns:
            historial_partido = historial_mlb[historial_mlb["match_id"].astype(str) == game_id].copy()
        else:
            historial_partido = pd.DataFrame()
        with st.expander("Historial y contratos Kalshi de este partido", expanded=False):
            if not historial_partido.empty:
                if "estado" not in historial_partido.columns:
                    historial_partido["estado"] = "PENDIENTE"
                historial_partido["estado"] = historial_partido["estado"].replace("", "PENDIENTE")
                cols = [x for x in [
                    "categoria", "mercado", "lado", "probabilidad_modelo", "precio_captura",
                    "edge", "estado", "marcador", "capturado_en"
                ] if x in historial_partido.columns]
                st.dataframe(
                    historial_partido[cols].sort_values("capturado_en", ascending=False)
                    if "capturado_en" in cols else historial_partido[cols],
                    use_container_width=True, hide_index=True
                )
                st.caption("Solo aparecen contratos capturados para este partido. WIN/LOSS proviene de la liquidación oficial de Kalshi.")
            else:
                st.info("Todavía no hay contratos de Kalshi archivados para este partido. Las proyecciones estadísticas de arriba son independientes del archivo de contratos.")

        # No mostramos el catálogo completo de líneas: solo las selecciones que pasan el filtro.

st.warning(
    f"El modelo mezcla producción de temporada y últimos 10 resultados, incorpora el ERA "
    f"medio de liga ({era_liga:.2f}) y el ERA del abridor probable cuando está publicado. "
    "Las probabilidades son estimaciones estadísticas, no garantías ni confirmación de valor frente a la cuota. "
    "Solo se enseñan selecciones que pasan filtros conservadores; aun así, antes de apostar hay que comprobar "
    "que exista la línea real, su precio y la alineación confirmada."
)
