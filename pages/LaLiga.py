from datetime import datetime, timedelta
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import re
from unicodedata import normalize
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import nbinom, poisson


API_FD = "https://api.football-data.org/v4"
API_KALSHI = "https://external-api.kalshi.com/trade-api/v2"
K = 6
N_GOLES = 11
N_CORNERS = 40
DISPERSION_CORNERS = 16
TZ = ZoneInfo("America/Chicago")

st.set_page_config(page_title="LaLiga · predicciones", page_icon="⚽", layout="wide")
st.title("🇪🇸 LaLiga · mercados y combinadas por partido")
st.caption(
    "Solo partidos de hoy y los próximos dos días. Elige un encuentro para ver el resumen "
    "de combinada limitado a mercados abiertos y líneas reales de Kalshi."
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


def _pedir_kalshi(path, **params):
    response = requests.get(f"{API_KALSHI}/{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=180, show_spinner=False)
def cargar_eventos_kalshi_laliga():
    """Trae eventos abiertos de todas las series cuyo ticker corresponde a LaLiga."""
    respuesta = _pedir_kalshi("series", limit=200, category="Sports")
    series = sorted({
        s["ticker"] for s in respuesta.get("series", [])
        if s.get("ticker", "").startswith("KXLALIGA")
        and "soccer" in [str(tag).lower() for tag in (s.get("tags") or [])]
    })
    if not series:
        raise ValueError("Kalshi no devolvió series de fútbol de LaLiga.")

    def traer_serie(ticker):
        eventos = []
        cursor = None
        for _ in range(5):
            params = {"limit": 200, "series_ticker": ticker, "status": "open",
                      "with_nested_markets": "true"}
            if cursor:
                params["cursor"] = cursor
            pagina = _pedir_kalshi("events", **params)
            eventos.extend(pagina.get("events", []))
            cursor = pagina.get("cursor")
            if not cursor:
                break
        return ticker, eventos

    eventos, errores = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        tareas = {pool.submit(traer_serie, ticker): ticker for ticker in series}
        for tarea in as_completed(tareas):
            ticker = tareas[tarea]
            try:
                _, lote = tarea.result()
                eventos.extend(lote)
            except requests.RequestException:
                errores.append(ticker)
    return eventos, errores, len(series)


def fecha_ticker_kalshi(evento):
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", evento.get("event_ticker", "").upper())
    if not match:
        return None
    meses = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
             "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    mes = meses.get(match.group(2))
    if not mes:
        return None
    try:
        return datetime(2000 + int(match.group(1)), mes, int(match.group(3))).date()
    except ValueError:
        return None


def clave_equipo_comparable(nombre):
    texto = limpiar(nombre)
    tokens = set(texto.split())
    # Nombres oficiales de football-data.org y Kalshi suelen llevar sufijos
    # diferentes; estos apodos son inequívocos dentro de LaLiga.
    if "espanyol" in tokens:
        return {"espanyol"}
    if "atletico" in tokens:
        return {"atletico"}
    if "athletic" in tokens or "bilbao" in tokens:
        return {"athletic"}
    if "vallecano" in tokens or "rayo" in tokens:
        return {"rayo"}
    if "betis" in tokens:
        return {"betis"}
    if "sociedad" in tokens:
        return {"sociedad"}
    if "alaves" in tokens:
        return {"alaves"}
    if "celta" in tokens:
        return {"celta"}
    if "madrid" in tokens and "real" in tokens:
        return {"real", "madrid"}
    ignorar = {"cf", "fc", "sc", "cd", "rcd", "ud", "de", "del", "la", "los", "las", "club"}
    return tokens - ignorar


def equipos_coinciden(nombre_a, nombre_b):
    a, b = limpiar(nombre_a), limpiar(nombre_b)
    if a == b:
        return True
    if min(len(a), len(b)) >= 8 and (a in b or b in a):
        return True
    ta, tb = clave_equipo_comparable(a), clave_equipo_comparable(b)
    return bool(ta and tb and len(ta & tb) / len(ta | tb) >= 0.8)


def buscar_equipo_en_texto(texto, local, visita):
    coincide_local = equipos_coinciden(local, texto)
    coincide_visita = equipos_coinciden(visita, texto)
    if coincide_local == coincide_visita:
        return None
    return local if coincide_local else visita


def evento_corresponde(evento, local, visita, fecha):
    fecha_kalshi = fecha_ticker_kalshi(evento)
    if fecha_kalshi != fecha:
        return False
    titulo = evento.get("title", "").split(":", 1)[0]
    equipos = re.split(r"\s+vs\.?\s+", titulo, maxsplit=1, flags=re.IGNORECASE)
    if len(equipos) != 2:
        return False
    return ((equipos_coinciden(equipos[0], local) and equipos_coinciden(equipos[1], visita))
            or (equipos_coinciden(equipos[0], visita) and equipos_coinciden(equipos[1], local)))


def cotizacion_valida(valor):
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if 0 < numero < 1 else None


def estimar_contrato_kalshi(evento, mercado, local, visita, p_goles, i, j,
                            p_corners, ci, cj):
    """Devuelve P(YES) solo cuando el contrato se puede mapear al modelo actual."""
    serie = evento.get("series_ticker", "")
    texto = " ".join(str(mercado.get(k, "")) for k in ("yes_sub_title", "title", "subtitle"))
    texto_limpio = limpiar(texto)
    linea_match = re.search(r"(\d+(?:\.\d+)?)", texto_limpio)
    try:
        linea = float(mercado.get("floor_strike"))
    except (TypeError, ValueError):
        linea = float(linea_match.group(1)) if linea_match else None

    condicion, dominio = None, "goles"
    if serie == "KXLALIGAGAME":
        lado = limpiar(mercado.get("yes_sub_title", ""))
        equipo = buscar_equipo_en_texto(lado, local, visita)
        if "tie" in lado or "draw" in lado or lado == "empate":
            condicion = i == j
        elif equipo == local:
            condicion = i > j
        elif equipo == visita:
            condicion = j > i
    elif serie == "KXLALIGASPREAD":
        equipo = buscar_equipo_en_texto(texto_limpio, local, visita)
        if equipo and linea is not None:
            margen = i - j if equipo == local else j - i
            condicion = margen > linea
    elif serie == "KXLALIGATOTAL" and linea is not None:
        condicion = (i + j > linea) if "over" in texto_limpio or "mas de" in texto_limpio else (i + j < linea)
    elif serie == "KXLALIGACORNERS" and p_corners is not None and ci is not None and cj is not None and linea is not None:
        condicion, dominio = ci + cj >= linea, "corners"
    elif serie == "KXLALIGATCORNERS" and p_corners is not None and ci is not None and cj is not None and linea is not None:
        equipo = buscar_equipo_en_texto(texto_limpio, local, visita)
        if equipo:
            condicion, dominio = (ci if equipo == local else cj) >= linea, "corners"
    elif serie == "KXLALIGATEAMTOTAL" and linea is not None:
        equipo = buscar_equipo_en_texto(texto_limpio, local, visita)
        if equipo:
            condicion = (i if equipo == local else j) > linea
    elif serie in {"KXLALIGABTTS"}:
        condicion = (i > 0) & (j > 0)
    elif serie in {"KXLALIGAFTTS", "KXLALIGAFIRSTGOAL"}:
        total = i + j
        equipo = buscar_equipo_en_texto(texto_limpio, local, visita)
        if "no goal" in texto_limpio or "sin gol" in texto_limpio:
            condicion = total == 0
        elif equipo == local:
            condicion = np.divide(i, total, out=np.zeros_like(p_goles), where=total > 0)
        elif equipo == visita:
            condicion = np.divide(j, total, out=np.zeros_like(p_goles), where=total > 0)
    elif serie == "KXLALIGASCORE":
        score = re.search(r"(\d+)\s*[-–]\s*(\d+)", texto_limpio)
        if score:
            a, b = int(score.group(1)), int(score.group(2))
            if "draw" in texto_limpio or "tie" in texto_limpio or "empate" in texto_limpio:
                condicion = (i == a) & (j == b)
            else:
                equipo = buscar_equipo_en_texto(texto_limpio, local, visita)
                if equipo == local:
                    condicion = (i == a) & (j == b)
                elif equipo == visita:
                    condicion = (i == b) & (j == a)

    if condicion is None:
        return None
    matriz = p_corners if dominio == "corners" else p_goles
    if matriz is None:
        return None
    return dominio, condicion, float(np.sum(matriz * np.asarray(condicion, dtype=float)))


def preparar_ofertas_kalshi(eventos, local, visita, fecha, p_goles, i, j,
                            p_corners, ci, cj, resultado=None, corners_final=None):
    ofertas, candidatos = [], []
    vistos = set()
    for evento in eventos:
        if not evento_corresponde(evento, local, visita, fecha):
            continue
        for mercado in evento.get("markets", []):
            if mercado.get("status") not in {None, "active", "open"}:
                continue
            ticker = mercado.get("ticker", "")
            if not ticker or ticker in vistos:
                continue
            vistos.add(ticker)
            estimacion = estimar_contrato_kalshi(
                evento, mercado, local, visita, p_goles, i, j, p_corners, ci, cj
            )
            prediccion_yes = estimacion[2] if estimacion else None
            yes_ask = cotizacion_valida(mercado.get("yes_ask_dollars"))
            no_ask = cotizacion_valida(mercado.get("no_ask_dollars"))
            etiqueta = mercado.get("yes_sub_title") or mercado.get("title") or ticker
            resultado_yes = None
            if estimacion and resultado is not None:
                dominio, condicion_yes, _ = estimacion
                if dominio == "goles":
                    serie = evento.get("series_ticker", "")
                    if serie in {"KXLALIGAFTTS", "KXLALIGAFIRSTGOAL"} and all(g > 0 for g in resultado):
                        resultado_yes = None
                    else:
                        resultado_yes = bool(condicion_yes[resultado[0], resultado[1]])
                elif dominio == "corners" and corners_final is not None:
                    resultado_yes = bool(condicion_yes[int(corners_final[0]), int(corners_final[1])])
            resultado_texto = (
                "Sin modelo" if estimacion is None else
                "Pendiente" if resultado_yes is None else
                "✅ Ganó YES" if resultado_yes else "✅ Ganó NO"
            )
            ofertas.append({
                "Mercado ofrecido por Kalshi": etiqueta,
                "Evento": evento.get("title", ""),
                "Precio YES": f"{yes_ask:.0%}" if yes_ask is not None else "Sin oferta",
                "Precio NO": f"{no_ask:.0%}" if no_ask is not None else "Sin oferta",
                "Prob. modelo YES": f"{prediccion_yes:.1%}" if prediccion_yes is not None else "Sin modelo",
                "Ventaja modelo vs YES": f"{prediccion_yes - yes_ask:+.1%}"
                    if prediccion_yes is not None and yes_ask is not None else "—",
                "Resultado": resultado_texto,
                "Ticker": ticker,
            })
            if estimacion:
                dominio, condicion_yes, p_yes = estimacion
                grupo = evento.get("series_ticker", "")
                for lado, precio, p_lado, cond in (
                    ("YES", yes_ask, p_yes, condicion_yes),
                    ("NO", no_ask, 1 - p_yes, 1 - np.asarray(condicion_yes, dtype=float)),
                ):
                    if precio is None or p_lado < 0.65 or p_lado - precio < 0.05:
                        continue
                    candidatos.append({
                        "mercado": f"{etiqueta} ({lado})", "grupo": grupo,
                        "evento": dominio, "condicion": cond, "p": p_lado,
                        "acierto": (resultado_yes if lado == "YES" else not resultado_yes)
                                   if resultado_yes is not None else None,
                        "precio": precio, "edge": p_lado - precio,
                    })
    return ofertas, candidatos


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

    def agregar(nombre, grupo, evento, condicion, probabilidad=None, tipo_resultado=None):
        matriz = p_corners if evento == "corners" else p
        if probabilidad is None:
            prob = float(np.sum(matriz * np.asarray(condicion, dtype=float)))
        else:
            prob = float(probabilidad)
        acierto = None
        if resultado is not None:
            goles_l, goles_v = resultado
            if tipo_resultado == "primer gol":
                acierto = None
            elif evento == "goles":
                acierto = bool(condicion[goles_l, goles_v])
            elif evento == "corners" and corners_final is not None and condicion is not None:
                acierto = bool(condicion[int(corners_final[0]), int(corners_final[1])])
        filas.append({"mercado": nombre, "grupo": grupo, "evento": evento, "tipo_resultado": tipo_resultado,
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

    # Probabilidad condicional del primer anotador dado el marcador final modelado.
    # Esto permite calcular combinadas coherentes con ganador, BTTS y total de goles.
    total_goles = i + j
    primero_local = np.divide(i, total_goles, out=np.zeros_like(p, dtype=float), where=total_goles > 0)
    primero_visita = np.divide(j, total_goles, out=np.zeros_like(p, dtype=float), where=total_goles > 0)
    sin_goles = total_goles == 0
    agregar(f"Primer gol: {local}", "Primer gol", "goles", primero_local,
            float(np.sum(p * primero_local)), "primer gol")
    agregar(f"Primer gol: {visita}", "Primer gol", "goles", primero_visita,
            float(np.sum(p * primero_visita)), "primer gol")
    agregar("Sin goles", "Primer gol", "goles", sin_goles, tipo_resultado="primer gol")

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
            if fila["tipo_resultado"] == "primer gol":
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
        matriz = p_goles if evento == "goles" else p_corners
        if matriz is None:
            return 0.0
        conjunta = np.prod(np.stack([np.asarray(x["condicion"], dtype=float) for x in sub]), axis=0)
        prob *= float(np.sum(matriz * conjunta))
    return prob


def resumir_combinadas(mercados, p_goles, p_corners):
    candidatos = [m for m in mercados if m["p"] >= 0.65]
    # El filtro se aplica a la probabilidad conjunta, no a la multiplicación
    # de probabilidades marginales. Exigimos al menos 50% para cualquier tamaño.
    umbral_conjunta = 0.50
    resumen = []
    for tamano in (2, 3, 4):
        opciones = []
        for legs in combinations(candidatos, tamano):
            grupos = [x["grupo"] for x in legs]
            if len(set(grupos)) != len(grupos):
                continue
            pc = probabilidad_combo(legs, p_goles, p_corners)
            if pc < umbral_conjunta:
                continue

            # Descarta selecciones prácticamente duplicadas (por ejemplo,
            # ganador y hándicap muy bajo que describen casi el mismo evento).
            redundante = False
            for a, b in combinations(legs, 2):
                if a["evento"] != b["evento"]:
                    continue
                prob_a = probabilidad_combo([a], p_goles, p_corners)
                prob_b = probabilidad_combo([b], p_goles, p_corners)
                conjunta_ab = probabilidad_combo([a, b], p_goles, p_corners)
                if min(prob_a, prob_b) > 0 and conjunta_ab / min(prob_a, prob_b) >= 0.98:
                    redundante = True
                    break
            if redundante:
                continue
            aciertos = [x["acierto"] for x in legs]
            resultado = False if any(x is False for x in aciertos) else True if all(x is True for x in aciertos) else None
            opciones.append((pc, legs, resultado))
        opciones.sort(key=lambda x: x[0], reverse=True)
        for pc, legs, resultado in opciones[:3]:
            estado = "Pendiente" if resultado is None else "✅ acertada" if resultado else "❌ fallada"
            resumen.append({"Tamaño": f"{tamano} selecciones",
                            "Combinada del partido": " + ".join(x["mercado"] for x in legs),
                            "Prob. conjunta": f"{pc:.1%}", "Resultado": estado})
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
    st.info("Kalshi puede ofrecer mercados de corners; sin estadísticas históricas suficientes, se mostrarán sus contratos pero el modelo no les asignará probabilidad.")

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

try:
    eventos_kalshi, series_kalshi_con_error, total_series_kalshi = cargar_eventos_kalshi_laliga()
    fecha_partido = datetime.fromisoformat(partido["utcDate"].replace("Z", "+00:00")).astimezone(TZ).date()
    ofertas_kalshi, candidatos_kalshi = preparar_ofertas_kalshi(
        eventos_kalshi, local, visita, fecha_partido, p_goles, i, j,
        p_corners, ci, cj, resultado, corners_final
    )
except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
    eventos_kalshi, series_kalshi_con_error, total_series_kalshi = [], [], 0
    ofertas_kalshi, candidatos_kalshi = [], []
    st.error(f"No pude consultar los mercados abiertos de Kalshi: {exc}")

combinadas = resumir_combinadas(candidatos_kalshi, p_goles, p_corners)
st.subheader(f"{local} vs {visita}")
if combinadas:
    st.markdown("### Resumen de combinadas con mercados reales de Kalshi")
    st.caption("Cada selección corresponde a un contrato abierto para este partido, supera 65% de probabilidad del modelo y tiene al menos 5 puntos porcentuales de ventaja estimada frente al precio YES/NO de compra de Kalshi. La probabilidad conjunta mínima es 50%; se muestran hasta tres combinadas por tamaño.")
    st.dataframe(pd.DataFrame(combinadas), use_container_width=True, hide_index=True)
else:
    st.info("No hay una combinada que cumpla los filtros usando contratos activos y líneas ofrecidas por Kalshi para este partido. No se inventan líneas ni se fuerzan sugerencias.")
st.write(f"**Marcador esperado:** {gl:.1f}–{gv:.1f} goles")

st.markdown("### Mercados realmente abiertos en Kalshi para este partido")
if ofertas_kalshi:
    df_ofertas = pd.DataFrame(ofertas_kalshi)
    st.dataframe(df_ofertas, use_container_width=True, hide_index=True)
else:
    if total_series_kalshi and not series_kalshi_con_error:
        st.info("Kalshi no tiene contratos abiertos para este encuentro y fecha. Por eso no se muestra una combinada.")
    else:
        st.info("No pude confirmar los contratos abiertos para este encuentro. No mostraré líneas del modelo como si fueran mercados de Kalshi.")

if series_kalshi_con_error:
    st.warning(f"Kalshi no respondió para {len(series_kalshi_con_error)} de {total_series_kalshi} series de LaLiga; el listado puede estar incompleto.")

st.caption(
    "El listado parte de los contratos activos que devuelve la API pública de Kalshi para el evento y fecha exactos. "
    "Los mercados que aún no tienen un modelo compatible se muestran como 'Sin modelo' y se excluyen de las combinadas."
)
st.warning(
    "La probabilidad y la ventaja son estimaciones del modelo, no garantías. Las combinadas que mezclan goles y corners "
    "suponen independencia entre esos datos. Revisa el contrato y las reglas en Kalshi antes de decidir."
)
