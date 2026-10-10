from datetime import datetime, timedelta
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil, floor
import re
import time
from pathlib import Path
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
HALF_LIFE_DAYS = 120
N_CORNERS = 40
DISPERSION_CORNERS = 16
TZ = ZoneInfo("America/Chicago")
PREDICCIONES_ARCHIVO = Path(__file__).resolve().parent / "predicciones_champions.csv"
RESULTADOS_ARCHIVO = Path(__file__).resolve().parent / "resultados_champions.csv"

def marcador_reglamentario(partido):
    """Use 90-minute score for Kalshi soccer markets, excluding extra time and penalties."""
    score = partido.get("score") or {}
    regular = score.get("regularTime") or {}
    if score.get("duration") in {"EXTRA_TIME", "PENALTY_SHOOTOUT"} and regular.get("home") is not None and regular.get("away") is not None:
        return regular
    return score.get("fullTime", {})

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
    actual = pedir_fd("competitions/CL/matches", clave, season=temporada)["matches"]
    anteriores = []
    try:
        anteriores = pedir_fd(
            "competitions/CL/matches", clave, season=temporada_anterior
        )["matches"]
    except requests.RequestException:
        pass
    historial = [m for m in anteriores + actual
                 if m.get("status") == "FINISHED"
                 and marcador_reglamentario(m).get("home") is not None]
    if not historial:
        raise ValueError("La API no devolvió partidos terminados para calcular el modelo.")

    ahora = datetime.now(TZ)
    estadisticas = calcular_estadisticas_goles(historial, ahora)
    return (actual, *estadisticas, ahora.date(), historial)


def calcular_estadisticas_goles(historial, fecha_corte):
    """Build weighted ratings from finished matches strictly before the cutoff."""
    validos = []
    for partido in historial:
        fecha_txt = partido.get("utcDate")
        if not fecha_txt:
            continue
        fecha = datetime.fromisoformat(fecha_txt.replace("Z", "+00:00"))
        if fecha >= fecha_corte or partido.get("status") != "FINISHED":
            continue
        marcador = marcador_reglamentario(partido)
        if marcador.get("home") is None or marcador.get("away") is None:
            continue
        validos.append((partido, fecha))
    if not validos:
        raise ValueError("No hay resultados anteriores al encuentro para calcular la predicción.")

    home_rows, away_rows = [], []
    forma = {}
    for m, fecha in validos:
        h = m["homeTeam"].get("shortName") or m["homeTeam"]["name"]
        a = m["awayTeam"].get("shortName") or m["awayTeam"]["name"]
        gh = int(marcador_reglamentario(m)["home"])
        ga = int(marcador_reglamentario(m)["away"])
        dias = max(0, (fecha_corte.astimezone(TZ) - fecha.astimezone(TZ)).days)
        # La mitad del peso desaparece cada cuatro meses para reflejar la forma reciente.
        peso = 0.5 ** (dias / HALF_LIFE_DAYS)
        home_rows.append((h, gh, ga, peso))
        away_rows.append((a, ga, gh, peso))
        forma.setdefault(h, []).append((fecha, gh, ga))
        forma.setdefault(a, []).append((fecha, ga, gh))

    home_df = pd.DataFrame(home_rows, columns=["equipo", "gf", "gc", "peso"])
    away_df = pd.DataFrame(away_rows, columns=["equipo", "gf", "gc", "peso"])
    prom_home = float(np.average(home_df["gf"], weights=home_df["peso"]))
    prom_away = float(np.average(away_df["gf"], weights=away_df["peso"]))
    if prom_home <= 0 or prom_away <= 0:
        raise ValueError("Faltan goles suficientes en el historial de Champions League.")

    home_df["gf_w"] = home_df.gf * home_df.peso
    home_df["gc_w"] = home_df.gc * home_df.peso
    away_df["gf_w"] = away_df.gf * away_df.peso
    away_df["gc_w"] = away_df.gc * away_df.peso
    home_stats = home_df.groupby("equipo").agg(gf=("gf_w", "sum"), gc=("gc_w", "sum"), n=("peso", "sum"))
    away_stats = away_df.groupby("equipo").agg(gf=("gf_w", "sum"), gc=("gc_w", "sum"), n=("peso", "sum"))
    home_attack = ((home_stats.gf + K * prom_home) / (home_stats.n + K)) / prom_home
    home_defence = ((home_stats.gc + K * prom_away) / (home_stats.n + K)) / prom_away
    away_attack = ((away_stats.gf + K * prom_away) / (away_stats.n + K)) / prom_away
    away_defence = ((away_stats.gc + K * prom_home) / (away_stats.n + K)) / prom_home
    return prom_home, prom_away, home_attack.to_dict(), home_defence.to_dict(), away_attack.to_dict(), away_defence.to_dict(), forma


def proporcion_goles_primer_tiempo(historial, fecha_corte):
    """Estimate the league's first-half goal share using only matches before kickoff."""
    suma_primera, suma_total = 0.0, 0.0
    for partido in historial:
        fecha_txt = partido.get("utcDate")
        if not fecha_txt or partido.get("status") != "FINISHED":
            continue
        fecha = datetime.fromisoformat(fecha_txt.replace("Z", "+00:00"))
        if fecha >= fecha_corte:
            continue
        half = partido.get("score", {}).get("halfTime", {})
        full = marcador_reglamentario(partido)
        if any(half.get(k) is None for k in ("home", "away")) or any(full.get(k) is None for k in ("home", "away")):
            continue
        goles_primera = int(half["home"]) + int(half["away"])
        goles_totales = int(full["home"]) + int(full["away"])
        dias = max(0, (fecha_corte.astimezone(TZ) - fecha.astimezone(TZ)).days)
        peso = 0.5 ** (dias / HALF_LIFE_DAYS)
        suma_primera += peso * goles_primera
        suma_total += peso * goles_totales
    if suma_total <= 0:
        return 0.45
    return float(np.clip(suma_primera / suma_total, 0.35, 0.65))


def limpiar(nombre):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode().lower()).split())


def _pedir_kalshi(path, **params):
    ultimo_error = None
    for intento in range(4):
        try:
            response = requests.get(f"{API_KALSHI}/{path}", params=params, timeout=25)
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            ultimo_error = exc
            if intento == 3:
                break
            time.sleep(1.5 * (2 ** intento))
    raise ultimo_error


@st.cache_data(ttl=180, show_spinner=False)
def cargar_eventos_kalshi_champions():
    """Trae eventos abiertos de todas las series cuyo ticker corresponde a Champions League."""
    series_disponibles, cursor, paginas = [], None, 0
    while paginas < 20:
        params = {"limit": 200, "category": "Sports"}
        if cursor:
            params["cursor"] = cursor
        respuesta = _pedir_kalshi("series", **params)
        series_disponibles.extend(respuesta.get("series", []))
        cursor = respuesta.get("cursor")
        paginas += 1
        if not cursor:
            break
    series = sorted({
        s["ticker"] for s in series_disponibles
        if s.get("ticker", "").startswith("KXUCL")
        and "soccer" in [str(tag).lower() for tag in (s.get("tags") or [])]
    })
    if not series:
        raise ValueError("Kalshi no devolvió series de fútbol de Champions League.")

    def traer_serie(ticker):
        eventos = []
        cursor = None
        paginas = 0
        while cursor is not None or paginas == 0:
            params = {"limit": 200, "series_ticker": ticker, "status": "open",
                      "with_nested_markets": "true"}
            if cursor:
                params["cursor"] = cursor
            pagina = _pedir_kalshi("events", **params)
            eventos.extend(pagina.get("events", []))
            cursor = pagina.get("cursor")
            paginas += 1
            if not cursor or paginas >= 50:
                break
        return ticker, eventos

    eventos, errores = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
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
    ignorar = {"cf", "fc", "sc", "cd", "rcd", "ud", "de", "del", "la", "los", "las", "club", "football", "calcio"}
    aliases = {
        "psg": "psg", "paris sg": "psg", "paris saint germain": "psg",
        "inter": "inter", "inter milan": "inter", "internazionale milano": "inter", "fc internazionale milano": "inter",
        "man city": "manchester city", "man utd": "manchester united", "man united": "manchester united",
        "sporting lisbon": "sporting cp", "sporting clube de portugal": "sporting cp",
        "bayern munchen": "bayern munich", "bayern": "bayern munich",
        "dortmund": "borussia dortmund", "bvb": "borussia dortmund",
        "leverkusen": "bayer leverkusen", "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur",
        "newcastle": "newcastle united", "psv": "psv eindhoven", "ajax": "ajax amsterdam",
        "club brugge": "brugge", "brujas": "brugge",
    }
    canonico = aliases.get(texto)
    tokens = set(canonico.split()) if canonico else set(texto.split())
    tokens -= ignorar
    if "psg" in tokens or {"paris", "saint", "germain"}.issubset(tokens):
        return {"psg"}
    if "inter" in tokens or "internazionale" in tokens:
        return {"inter"}
    if "bvb" in tokens or {"borussia", "dortmund"}.issubset(tokens):
        return {"dortmund"}
    if "bayern" in tokens:
        return {"bayern"}
    if "psv" in tokens:
        return {"psv"}
    if "ajax" in tokens:
        return {"ajax"}
    if "salzburg" in tokens:
        return {"salzburg"}
    return tokens

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


def tarifa_estimada_kalshi(precio):
    """Estimate one standard taker fee; market-specific rates may differ."""
    return ceil(7 * precio * (1 - precio)) / 100


def precio_maximo_para_margen(probabilidad, ganancia_minima=0.05, roi_minimo=0.10):
    """Highest cent price meeting both net expected profit and ROI targets."""
    if probabilidad < 0.50:
        return None
    validos = []
    for centavos in range(1, 100):
        precio = centavos / 100
        costo = precio + tarifa_estimada_kalshi(precio)
        ev = probabilidad - costo
        roi = ev / costo if costo else 0
        if ev >= ganancia_minima and roi >= roi_minimo:
            validos.append(precio)
    return max(validos) if validos else None


def _texto_mercado(mercado):
    return limpiar(" ".join(str(mercado.get(k, "")) for k in ("yes_sub_title", "title", "subtitle")))


def _linea_mercado(mercado):
    match = re.search(r"(\d+(?:\.\d+)?)", _texto_mercado(mercado))
    if match:
        return float(match.group(1))
    try:
        return float(mercado.get("floor_strike"))
    except (TypeError, ValueError):
        return None


def _direccion_mercado(mercado):
    texto = _texto_mercado(mercado)
    if any(palabra in texto for palabra in ("under", "less than", "fewer than", "menos de", "menos que", "below")):
        return "under"
    return "over"


def clave_plantilla_kalshi(evento, mercado):
    """Tipo y línea observados en un contrato real de Champions League, sin depender de equipos."""
    serie = evento.get("series_ticker", "")
    series_con_linea = {
        "KXUCLSPREAD", "KXUCLTOTAL", "KXUCLCORNERS",
        "KXUCLTCORNERS", "KXUCLTEAMTOTAL", "KXUCL1HSPREAD",
        "KXUCL1HTOTAL", "KXUCL2HTOTAL",
    }
    series_sin_linea = {
        "KXUCLGAME", "KXUCLBTTS", "KXUCLFTTS", "KXUCLFIRSTGOAL",
        "KXUCL1H", "KXUCL1HBTTS", "KXUCL2H",
    }
    if serie in series_sin_linea:
        return serie, None, None
    linea = _linea_mercado(mercado)
    if serie in series_con_linea and linea is not None:
        direccion = _direccion_mercado(mercado) if serie in {
            "KXUCLTOTAL", "KXUCLCORNERS", "KXUCLTCORNERS", "KXUCLTEAMTOTAL",
            "KXUCL1HTOTAL", "KXUCL2HTOTAL",
        } else None
        return serie, linea, direccion
    return None


def clave_contrato_kalshi(evento, mercado, local, visita):
    """Clave del resultado YES exacto, relativo al partido seleccionado."""
    plantilla = clave_plantilla_kalshi(evento, mercado)
    if plantilla is None:
        return None
    serie, linea, direccion = plantilla
    texto = _texto_mercado(mercado)
    if serie in {"KXUCLGAME", "KXUCL1H", "KXUCL2H"}:
        lado = limpiar(mercado.get("yes_sub_title") or mercado.get("title", ""))
        if "tie" in lado or "draw" in lado or lado == "empate":
            return serie, "empate", None, None
        equipo = buscar_equipo_en_texto(lado, local, visita)
        return (serie, "local" if equipo == local else "visita", None, None) if equipo else None
    if serie in {"KXUCLSPREAD", "KXUCL1HSPREAD"}:
        equipo = buscar_equipo_en_texto(texto, local, visita)
        return (serie, "local" if equipo == local else "visita", linea, direccion) if equipo else None
    if serie in {"KXUCLTOTAL", "KXUCLCORNERS", "KXUCL1HTOTAL", "KXUCL2HTOTAL"}:
        return serie, "over", linea, direccion
    if serie in {"KXUCLTCORNERS", "KXUCLTEAMTOTAL"}:
        equipo = buscar_equipo_en_texto(texto, local, visita)
        return (serie, "local" if equipo == local else "visita", linea, direccion) if equipo else None
    if serie in {"KXUCLBTTS", "KXUCL1HBTTS"}:
        return serie, "ambos", None, None
    if serie in {"KXUCLFTTS", "KXUCLFIRSTGOAL"}:
        if "no goal" in texto or "sin gol" in texto:
            return serie, "sin_goles", None, None
        equipo = buscar_equipo_en_texto(texto, local, visita)
        return (serie, "local" if equipo == local else "visita", None, None) if equipo else None
    return None


def modelar_seleccion_kalshi(clave, local, visita, p_goles, i, j, p_corners, ci, cj, proporcion_primera=0.45):
    serie, lado, linea, direccion = clave
    dominio, condicion = "goles", None
    periodo = proporcion_primera if serie.startswith("KXUCL1H") else 1 - proporcion_primera if serie.startswith("KXUCL2H") else None
    matriz_goles = p_goles
    if periodo is not None:
        goles = np.arange(p_goles.shape[0])
        lambda_local = float(np.sum(p_goles * i))
        lambda_visita = float(np.sum(p_goles * j))
        matriz_goles = np.outer(
            poisson.pmf(goles, lambda_local * periodo),
            poisson.pmf(goles, lambda_visita * periodo),
        )
        matriz_goles /= matriz_goles.sum()
        i, j = np.meshgrid(goles, goles, indexing="ij")
    if serie in {"KXUCLGAME", "KXUCL1H", "KXUCL2H"}:
        condicion = i == j if lado == "empate" else i > j if lado == "local" else j > i
    elif serie in {"KXUCLSPREAD", "KXUCL1HSPREAD"} and linea is not None:
        margen = i - j if lado == "local" else j - i
        condicion = margen > linea
    elif serie in {"KXUCLTOTAL", "KXUCL1HTOTAL", "KXUCL2HTOTAL"} and linea is not None:
        condicion = i + j < linea if direccion == "under" else i + j > linea
    elif serie == "KXUCLCORNERS" and p_corners is not None and ci is not None and cj is not None and linea is not None:
        dominio, condicion = "corners", ci + cj < linea if direccion == "under" else ci + cj >= linea
    elif serie == "KXUCLTCORNERS" and p_corners is not None and ci is not None and cj is not None and linea is not None:
        dominio, condicion = "corners", (ci if lado == "local" else cj) < linea if direccion == "under" else (ci if lado == "local" else cj) >= linea
    elif serie == "KXUCLTEAMTOTAL" and linea is not None:
        condicion = (i if lado == "local" else j) < linea if direccion == "under" else (i if lado == "local" else j) > linea
    elif serie in {"KXUCLBTTS", "KXUCL1HBTTS"}:
        condicion = (i > 0) & (j > 0)
    elif serie in {"KXUCLFTTS", "KXUCLFIRSTGOAL"}:
        total = i + j
        if lado == "sin_goles":
            condicion = total == 0
        elif lado == "local":
            condicion = np.divide(i, total, out=np.zeros_like(p_goles), where=total > 0)
        else:
            condicion = np.divide(j, total, out=np.zeros_like(p_goles), where=total > 0)
    if condicion is None:
        return None
    matriz = p_corners if dominio == "corners" else matriz_goles
    if matriz is None:
        return None
    probabilidad = float(np.sum(matriz * np.asarray(condicion, dtype=float)))
    return dominio, condicion, probabilidad


def resultado_del_periodo(clave, resultados):
    """Return the actual score for the same period as a Kalshi market."""
    serie = clave[0]
    if serie.startswith("KXUCL1H"):
        return resultados.get("primer_tiempo")
    if serie.startswith("KXUCL2H"):
        return resultados.get("segundo_tiempo")
    return resultados.get("partido")


def evaluar_yes_goles(clave, marcador):
    """Settle goal markets directly from their period score; None means unknowable."""
    if marcador is None:
        return None
    serie, lado, linea, direccion = clave
    local, visita = marcador
    total = local + visita
    if serie in {"KXUCLGAME", "KXUCL1H", "KXUCL2H"}:
        return local == visita if lado == "empate" else local > visita if lado == "local" else visita > local
    if serie in {"KXUCLSPREAD", "KXUCL1HSPREAD"}:
        diferencia = local - visita if lado == "local" else visita - local
        return diferencia > linea
    if serie in {"KXUCLTOTAL", "KXUCL1HTOTAL", "KXUCL2HTOTAL"}:
        return total < linea if direccion == "under" else total > linea
    if serie == "KXUCLTEAMTOTAL":
        goles_equipo = local if lado == "local" else visita
        return goles_equipo < linea if direccion == "under" else goles_equipo > linea
    if serie in {"KXUCLBTTS", "KXUCL1HBTTS"}:
        return local > 0 and visita > 0
    if serie in {"KXUCLFTTS", "KXUCLFIRSTGOAL"}:
        if lado == "sin_goles":
            return total == 0
        if total == 0 or (local > 0 and visita > 0):
            return None
        return local > 0 if lado == "local" else visita > 0
    return None


def etiqueta_seleccion_kalshi(clave, local, visita):
    serie, lado, linea, direccion = clave
    equipo = local if lado == "local" else visita
    periodo = " (1.ª parte)" if serie.startswith("KXUCL1H") else " (2.ª parte)" if serie.startswith("KXUCL2H") else ""
    if serie in {"KXUCLGAME", "KXUCL1H", "KXUCL2H"}:
        return ("Empate" if lado == "empate" else f"Gana {equipo}") + periodo
    if serie in {"KXUCLSPREAD", "KXUCL1HSPREAD"}:
        return f"{equipo} gana por más de {linea:g} goles{periodo}"
    if serie in {"KXUCLTOTAL", "KXUCL1HTOTAL", "KXUCL2HTOTAL"}:
        return (f"Menos de {linea:g} goles totales{periodo}" if direccion == "under"
                else f"Más de {linea:g} goles totales{periodo}")
    if serie == "KXUCLCORNERS":
        return f"Menos de {linea:g} corners totales" if direccion == "under" else f"{linea:g}+ corners totales"
    if serie == "KXUCLTCORNERS":
        return f"Menos de {linea:g} corners de {equipo}" if direccion == "under" else f"{linea:g}+ corners de {equipo}"
    if serie == "KXUCLTEAMTOTAL":
        return f"Menos de {linea:g} goles de {equipo}" if direccion == "under" else f"Más de {linea:g} goles de {equipo}"
    if serie in {"KXUCLBTTS", "KXUCL1HBTTS"}:
        return "Ambos equipos marcan" + periodo
    if lado == "sin_goles":
        return "No se marca ningún gol"
    return f"Primer gol de {equipo}"


def etiqueta_no_kalshi(clave, local, visita):
    serie, lado, linea, direccion = clave
    periodo = " (1.ª parte)" if serie.startswith("KXUCL1H") else " (2.ª parte)" if serie.startswith("KXUCL2H") else ""
    if serie in {"KXUCLGAME", "KXUCL1H", "KXUCL2H"}:
        otro = visita if lado == "local" else local
        return f"No gana {local if lado == 'local' else visita} (empate o gana {otro})" if lado != "empate" else "No hay empate (gana local o visita)"
    if serie in {"KXUCLTOTAL", "KXUCL1HTOTAL", "KXUCL2HTOTAL"}:
        opuesta = "Más de" if direccion == "under" else "Menos de"
        return f"NO: {opuesta} {linea:g} goles totales{periodo}"
    if serie == "KXUCLCORNERS":
        opuesta = f"{linea:g}+" if direccion == "under" else f"Menos de {linea:g}"
        return f"NO: {opuesta} corners totales"
    if serie == "KXUCLTCORNERS":
        opuesta = f"{linea:g}+" if direccion == "under" else f"Menos de {linea:g}"
        return f"NO: {opuesta} corners de {local if lado == 'local' else visita}"
    if serie == "KXUCLTEAMTOTAL":
        opuesta = "Más de" if direccion == "under" else "Menos de"
        return f"NO: {opuesta} {linea:g} goles de {local if lado == 'local' else visita}{periodo}"
    if serie in {"KXUCLBTTS", "KXUCL1HBTTS"}:
        return "Al menos un equipo no marca" + periodo
    if serie in {"KXUCLFTTS", "KXUCLFIRSTGOAL"}:
        return "El primer gol no es de " + ("nadie" if lado == "sin_goles" else local if lado == "local" else visita)
    return f"NO: {etiqueta_seleccion_kalshi(clave, local, visita)}"


def categoria_resumen_sencillo(clave):
    serie = clave[0]
    return {
        "KXUCLGAME": "Resultado",
        "KXUCL1H": "Resultado 1.ª parte",
        "KXUCL2H": "Resultado 2.ª parte",
        "KXUCLSPREAD": "Hándicap de goles",
        "KXUCL1HSPREAD": "Hándicap 1.ª parte",
        "KXUCLTOTAL": "Goles totales",
        "KXUCL1HTOTAL": "Goles 1.ª parte",
        "KXUCL2HTOTAL": "Goles 2.ª parte",
        "KXUCLCORNERS": "Corners totales",
        "KXUCLTCORNERS": "Corners por equipo",
        "KXUCLTEAMTOTAL": "Goles por equipo",
        "KXUCLBTTS": "Ambos marcan",
        "KXUCL1HBTTS": "Ambos marcan 1.ª parte",
        "KXUCLFTTS": "Primer gol",
        "KXUCLFIRSTGOAL": "Primer gol",
    }.get(serie)


def etiqueta_resumen_sencillo(clave, lado_apuesta, local, visita):
    """Affirmative, plain-language phrasing for the match overview."""
    serie, lado, linea, direccion = clave
    yes = lado_apuesta == "YES"
    if serie in {"KXUCLGAME", "KXUCL1H", "KXUCL2H"}:
        periodo = " (1.ª parte)" if serie == "KXUCL1H" else " (2.ª parte)" if serie == "KXUCL2H" else ""
        if yes:
            return (f"Gana {local}" if lado == "local" else f"Gana {visita}" if lado == "visita" else "Empate") + periodo
        if lado == "local":
            return f"Visita o empate (X2){periodo}"
        if lado == "visita":
            return f"Local o empate (1X){periodo}"
        return f"Gana {local} o {visita}{periodo}"

    if serie in {"KXUCLSPREAD", "KXUCL1HSPREAD"}:
        equipo = local if lado == "local" else visita
        contrario = visita if lado == "local" else local
        periodo = " (1.ª parte)" if serie == "KXUCL1HSPREAD" else ""
        minimo = floor(linea) + 1
        unidad = "gol" if linea == 1 else "goles"
        return (f"{equipo} gana por {minimo}+ goles{periodo}" if yes
                else f"{contrario} no pierde por más de {linea:g} {unidad}{periodo}")

    if serie in {"KXUCLTOTAL", "KXUCL1HTOTAL", "KXUCL2HTOTAL", "KXUCLCORNERS", "KXUCLTCORNERS", "KXUCLTEAMTOTAL"}:
        yes_is_over = direccion != "under"
        es_over = yes == yes_is_over
        periodo = " (1.ª parte)" if serie == "KXUCL1HTOTAL" else " (2.ª parte)" if serie == "KXUCL2HTOTAL" else ""
        if serie == "KXUCLTOTAL":
            if es_over:
                return f"Más de {linea:g} goles totales"
            maximo = max(0, ceil(linea) - 1)
            return f"0–{maximo} goles totales"
        if serie in {"KXUCL1HTOTAL", "KXUCL2HTOTAL"}:
            return (f"Más de {linea:g} goles{periodo}" if es_over
                    else f"0–{max(0, ceil(linea) - 1)} goles{periodo}")
        if serie == "KXUCLCORNERS":
            if es_over:
                return f"{ceil(linea)}+ corners totales"
            maximo = max(0, ceil(linea) - 1)
            return f"0–{maximo} corners totales"
        equipo = local if lado == "local" else visita
        if serie == "KXUCLTCORNERS":
            if es_over:
                return f"{ceil(linea)}+ corners de {equipo}"
            maximo = max(0, ceil(linea) - 1)
            return f"0–{maximo} corners de {equipo}"
        if es_over:
            return f"Más de {linea:g} goles de {equipo}"
        maximo = max(0, ceil(linea) - 1)
        return f"{maximo} o menos goles de {equipo}"

    if serie in {"KXUCLBTTS", "KXUCL1HBTTS"}:
        periodo = " en la 1.ª parte" if serie == "KXUCL1HBTTS" else ""
        return ("Ambos equipos marcan" if yes else "Uno o ambos equipos se quedan sin marcar") + periodo
    if serie in {"KXUCLFTTS", "KXUCLFIRSTGOAL"}:
        if yes:
            return "No habrá goles" if lado == "sin_goles" else f"Primer gol de {local if lado == 'local' else visita}"
        if lado == "sin_goles":
            return "Habrá al menos un gol"
        otro = visita if lado == "local" else local
        return f"Primer gol de {otro} o sin goles"
    return etiqueta_seleccion_kalshi(clave, local, visita) if yes else etiqueta_no_kalshi(clave, local, visita)


def preparar_top_predicciones_kalshi(eventos, local, visita, fecha, p_goles, i, j,
                                    p_corners, ci, cj, resultado=None, corners_final=None, proporcion_primera=0.45):
    plantillas, ofertas_actuales, mercados_actuales = set(), {}, []
    for evento in eventos:
        for mercado in evento.get("markets", []):
            if mercado.get("status") not in {None, "active", "open"}:
                continue
            plantilla = clave_plantilla_kalshi(evento, mercado)
            if plantilla:
                plantillas.add(plantilla)
            if evento_corresponde(evento, local, visita, fecha):
                clave = clave_contrato_kalshi(evento, mercado, local, visita)
                estimacion_actual = (
                    modelar_seleccion_kalshi(
                        clave, local, visita, p_goles, i, j, p_corners, ci, cj, proporcion_primera
                    )
                    if clave else None
                )
                prob_yes_actual = estimacion_actual[2] if estimacion_actual else None
                precio_yes_actual = cotizacion_valida(mercado.get("yes_ask_dollars"))
                precio_no_actual = cotizacion_valida(mercado.get("no_ask_dollars"))
                mercados_actuales.append({
                    "Mercado Kalshi": mercado.get("yes_sub_title") or mercado.get("title") or mercado.get("ticker"),
                    "Prob. YES modelo": f"{prob_yes_actual:.1%}" if prob_yes_actual is not None else "Sin modelo para este tipo",
                    "Precio YES": f"{precio_yes_actual:.0%}" if precio_yes_actual is not None else "Sin oferta",
                    "Precio NO": f"{precio_no_actual:.0%}" if precio_no_actual is not None else "Sin oferta",
                    "Reglas del contrato": mercado.get("rules_primary") or mercado.get("rules_secondary") or "Consultar en Kalshi",
                    "Ticker": mercado.get("ticker", ""),
                })
                if clave:
                    oferta_previa = ofertas_actuales.get(clave)
                    if oferta_previa is None or (
                        cotizacion_valida(mercado.get("yes_ask_dollars")) is not None
                        and cotizacion_valida(oferta_previa.get("yes_ask_dollars")) is None
                    ):
                        ofertas_actuales[clave] = mercado

    claves = set()
    for serie, linea, direccion in plantillas:
        if serie in {"KXUCLGAME", "KXUCL1H", "KXUCL2H"}:
            claves.update({(serie, "local", None, None), (serie, "empate", None, None), (serie, "visita", None, None)})
        elif serie in {"KXUCLSPREAD", "KXUCLTCORNERS", "KXUCLTEAMTOTAL", "KXUCL1HSPREAD"}:
            claves.update({(serie, "local", linea, direccion), (serie, "visita", linea, direccion)})
        elif serie in {"KXUCLFTTS", "KXUCLFIRSTGOAL"}:
            claves.update({(serie, "local", None, None), (serie, "visita", None, None), (serie, "sin_goles", None, None)})
        elif serie in {"KXUCLBTTS", "KXUCL1HBTTS"}:
            claves.add((serie, "ambos", None, None))
        else:
            claves.add((serie, "over", linea, direccion))

    predicciones, oportunidades_margen, candidatos_cotizados = [], [], []
    for clave in claves:
        estimacion = modelar_seleccion_kalshi(clave, local, visita, p_goles, i, j, p_corners, ci, cj, proporcion_primera)
        if estimacion is None:
            continue
        dominio, condicion_yes, probabilidad_yes = estimacion
        mercado = ofertas_actuales.get(clave)
        for lado_apuesta in ("YES", "NO"):
            probabilidad = probabilidad_yes if lado_apuesta == "YES" else 1 - probabilidad_yes
            condicion = condicion_yes if lado_apuesta == "YES" else 1 - np.asarray(condicion_yes, dtype=float)
            precio_campo = "yes_ask_dollars" if lado_apuesta == "YES" else "no_ask_dollars"
            precio = cotizacion_valida(mercado.get(precio_campo)) if mercado else None
            precio_objetivo = precio_maximo_para_margen(probabilidad)
            if precio is not None:
                tarifa = tarifa_estimada_kalshi(precio)
                costo_total = precio + tarifa
                ganancia_si_acierta = 1 - costo_total
                ganancia_esperada = probabilidad - costo_total
                roi_esperado = ganancia_esperada / costo_total if costo_total > 0 else -1
                supera_filtros = probabilidad >= 0.50 and ganancia_esperada >= 0.05 and roi_esperado >= 0.10
                candidatos_cotizados.append({
                    "Apuesta": etiqueta_seleccion_kalshi(clave, local, visita) if lado_apuesta == "YES" else etiqueta_no_kalshi(clave, local, visita),
                    "Lado": lado_apuesta,
                    "Prob. modelo": f"{probabilidad:.1%}",
                    "Precio actual": f"${precio:.2f}",
                    "EV neto por contrato": f"${ganancia_esperada:+.2f}",
                    "ROI neto": f"{roi_esperado:+.1%}",
                    "Precio máx. para margen objetivo": f"${precio_objetivo:.2f}" if precio_objetivo else "No alcanza el mínimo",
                    "Evaluación": "Cumple" if supera_filtros else "No tiene margen suficiente al precio actual",
                    "Ticker": mercado.get("ticker", ""),
                    "_ev": ganancia_esperada,
                    "_roi": roi_esperado,
                })
                # Minimum filters prevent presenting tiny theoretical edges as useful bets.
                if supera_filtros:
                    oportunidades_margen.append({
                        "Apuesta": etiqueta_seleccion_kalshi(clave, local, visita) if lado_apuesta == "YES" else etiqueta_no_kalshi(clave, local, visita),
                        "Lado": lado_apuesta,
                        "Probabilidad del modelo": f"{probabilidad:.1%}",
                        "Precio Kalshi": f"${precio:.2f}",
                        "Tarifa estimada": f"${tarifa:.2f}",
                        "Ganancia si acierta (1 contrato)": f"${ganancia_si_acierta:.2f}",
                        "Ganancia esperada neta": f"${ganancia_esperada:+.2f}",
                        "ROI neto esperado": f"{roi_esperado:.1%}",
                        "Ticker": mercado.get("ticker", ""),
                        "_roi": roi_esperado,
                        "_ev": ganancia_esperada,
                    })
            estado = (
                "Disponible ahora" if mercado and precio is not None else
                f"Kalshi lista el contrato, sin oferta {lado_apuesta}" if mercado else
                "Tipo/línea ofrecido en Champions League; falta abrirlo para este partido"
            )

            resultado_texto = (
                "Pendiente de datos de corners"
                if resultado is not None and dominio == "corners" and corners_final is None
                else "Pendiente"
            )
            if resultado is not None and dominio == "goles":
                marcador_periodo = resultado_del_periodo(clave, resultado)
                acierto_yes = evaluar_yes_goles(clave, marcador_periodo)
                if acierto_yes is None and clave[0] in {"KXUCLFTTS", "KXUCLFIRSTGOAL"}:
                    resultado_texto = "Sin dato de primer anotador"
                elif acierto_yes is None and marcador_periodo is None:
                    resultado_texto = "Sin dato del descanso"
                elif acierto_yes is not None:
                    acierto = acierto_yes if lado_apuesta == "YES" else not acierto_yes
                    resultado_texto = "✅ Se cumplió" if acierto else "❌ No se cumplió"
            elif resultado is not None and corners_final is not None and dominio == "corners":
                serie, lado, linea, direccion = clave
                cuenta = sum(corners_final) if serie == "KXUCLCORNERS" else corners_final[0] if lado == "local" else corners_final[1]
                acierto_yes = cuenta < linea if direccion == "under" else cuenta >= linea
                acierto = acierto_yes if lado_apuesta == "YES" else not acierto_yes
                resultado_texto = "✅ Se cumplió" if acierto else "❌ No se cumplió"

            predicciones.append({
                "Apuesta": etiqueta_seleccion_kalshi(clave, local, visita) if lado_apuesta == "YES" else etiqueta_no_kalshi(clave, local, visita),
                "Apuesta sencilla": etiqueta_resumen_sencillo(clave, lado_apuesta, local, visita),
                "Categoría resumen": categoria_resumen_sencillo(clave),
                "Lado Kalshi": lado_apuesta,
                "Probabilidad del modelo": f"{probabilidad:.1%}",
                "Precio de compra": f"{precio:.0%}" if precio is not None else "—",
                "Precio máx. para margen objetivo": f"${precio_objetivo:.2f}" if precio_objetivo else "—",
                "Disponibilidad": estado,
                "Ticker": mercado.get("ticker", "") if mercado else "—",
                "Resultado": resultado_texto,
                "_clave": clave,
                "_lado": lado_apuesta,
                "_p": probabilidad,
            })

    predicciones.sort(key=lambda x: x["_p"], reverse=True)
    candidatos_resumen = {}
    for fila in predicciones:
        categoria = fila["Categoría resumen"]
        if categoria in {"Resultado", "Resultado 1.ª parte", "Resultado 2.ª parte"} and fila["Lado Kalshi"] == "NO" and "empate" in fila["Apuesta"].lower():
            # Prefer double-chance phrasing (1X/X2) over the less readable no-draw outcome.
            continue
        if categoria:
            candidatos_resumen.setdefault(categoria, []).append(fila)
    resumen_por_categoria = {}
    for categoria, filas in candidatos_resumen.items():
        # Prefer useful mid-range probabilities to ultra-short lines with negligible payout.
        rango_util = [fila for fila in filas if 0.55 <= fila["_p"] <= 0.80]
        mejor = max(rango_util or filas, key=lambda fila: fila["_p"])
        resumen_por_categoria[categoria] = {
            "Tipo": categoria,
            "Jugada sencilla": mejor["Apuesta sencilla"],
            "Probabilidad modelo": mejor["Probabilidad del modelo"],
            "Precio ahora": mejor["Precio de compra"],
            "Disponibilidad": mejor["Disponibilidad"],
            "Ticker": mejor["Ticker"],
            "Resultado": mejor["Resultado"],
            "Probabilidad base": mejor["_p"],
            "Clave interna": repr((mejor["_clave"], mejor["_lado"])),
            "_p": mejor["_p"],
        }
    resumen_simple = sorted(resumen_por_categoria.values(), key=lambda fila: fila["_p"], reverse=True)[:10]
    resumen_simple = [{k: v for k, v in fila.items() if k != "_p"} for fila in resumen_simple]
    todas = [{k: v for k, v in fila.items() if k != "_p"} for fila in predicciones]
    top = todas[:10]
    oportunidades_margen.sort(key=lambda x: (x["_roi"], x["_ev"]), reverse=True)
    oportunidades_margen = [
        {k: v for k, v in fila.items() if not k.startswith("_")}
        for fila in oportunidades_margen
    ]
    margen_yes = [x for x in oportunidades_margen if x["Lado"] == "YES"][:5]
    margen_no = [x for x in oportunidades_margen if x["Lado"] == "NO"][:5]
    candidatos_cotizados.sort(key=lambda x: (x["_ev"], x["_roi"]), reverse=True)
    candidatos_yes = [x for x in candidatos_cotizados if x["Lado"] == "YES"][:5]
    candidatos_no = [x for x in candidatos_cotizados if x["Lado"] == "NO"][:5]
    candidatos_yes = [{k: v for k, v in x.items() if not k.startswith("_")} for x in candidatos_yes]
    candidatos_no = [{k: v for k, v in x.items() if not k.startswith("_")} for x in candidatos_no]
    mercados_actuales.sort(key=lambda x: x["Mercado Kalshi"])
    return resumen_simple, margen_yes, margen_no, candidatos_yes, candidatos_no, top, todas, mercados_actuales, len(plantillas)


@st.cache_data(ttl=21600, show_spinner=False)
def cargar_corners(temporadas):
    """La API usada aquí no aporta corners históricos de Champions League."""
    return None
def media_corners_ponderada(registros, promedio_liga):
    peso_total = sum(peso for _, peso in registros)
    valor_total = sum(valor * peso for valor, peso in registros)
    return (valor_total + K * promedio_liga) / (peso_total + K)


def modelo_corners_antes_de_fecha(datos, fecha_corte):
    """Corner ratings from dated matches before the fixture's local calendar day."""
    partidos = datos.get("partidos")
    if partidos is None or partidos.empty:
        return None
    corte = pd.Timestamp(fecha_corte.astimezone(TZ).date())
    anteriores = partidos[partidos["MatchDate"].notna() & (partidos["MatchDate"] < corte)].copy()
    if anteriores.empty:
        return None
    dias = (corte - anteriores["MatchDate"]).dt.days.clip(lower=0)
    anteriores["peso"] = 0.5 ** (dias / HALF_LIFE_DAYS)
    equipos = {}
    for r in anteriores.itertuples(index=False):
        h, a = limpiar(r.HomeTeam), limpiar(r.AwayTeam)
        equipos.setdefault(h, {"home_for": [], "home_against": [], "away_for": [], "away_against": []})
        equipos.setdefault(a, {"home_for": [], "home_against": [], "away_for": [], "away_against": []})
        equipos[h]["home_for"].append((float(r.HC), float(r.peso)))
        equipos[h]["home_against"].append((float(r.AC), float(r.peso)))
        equipos[a]["away_for"].append((float(r.AC), float(r.peso)))
        equipos[a]["away_against"].append((float(r.HC), float(r.peso)))
    return {
        "equipos": equipos,
        "prom_home": float(np.average(anteriores.HC, weights=anteriores.peso)),
        "prom_away": float(np.average(anteriores.AC, weights=anteriores.peso)),
        "nombres": list(equipos),
    }


def resultado_corners(datos, local, visita, fecha=None):
    partidos = datos.get("partidos")
    if fecha is not None and partidos is not None and not partidos.empty:
        objetivo = fecha.astimezone(TZ).date() if isinstance(fecha, datetime) else fecha
        candidatos = []
        for fila in partidos.itertuples(index=False):
            if pd.isna(fila.MatchDate) or not equipos_coinciden(fila.HomeTeam, local) or not equipos_coinciden(fila.AwayTeam, visita):
                continue
            diferencia = abs((fila.MatchDate.date() - objetivo).days)
            if diferencia <= 1 and pd.notna(fila.HC) and pd.notna(fila.AC):
                candidatos.append((diferencia, (float(fila.HC), float(fila.AC))))
        if candidatos:
            return min(candidatos, key=lambda item: item[0])[1]
        return None
    for (home, away), total in datos.get("resultados", {}).items():
        if equipos_coinciden(home, local) and equipos_coinciden(away, visita):
            return total
    return None


def clave_equipo(nombre, nombres):
    exacta = limpiar(nombre)
    if exacta in nombres:
        return exacta
    match = get_close_matches(exacta, nombres, n=1, cutoff=0.58)
    return match[0] if match else None


def estimar_goles(local, visita, modelo, fecha_partido=None):
    _, ph, pa, ha, hd, aa, ad, formas, _, _ = modelo
    lh = ph * ha.get(local, 1.0) * ad.get(visita, 1.0)
    av = pa * aa.get(visita, 1.0) * hd.get(local, 1.0)
    if fecha_partido is None:
        fecha_corte = datetime.now(TZ)
    elif isinstance(fecha_partido, datetime):
        fecha_corte = fecha_partido.astimezone(TZ)
    else:
        fecha_corte = datetime.combine(fecha_partido, datetime.min.time(), tzinfo=TZ)

    def factores_forma(equipo):
        recientes = sorted(
            (r for r in formas.get(equipo, []) if r[0].astimezone(TZ) < fecha_corte),
            key=lambda r: r[0], reverse=True,
        )[:8]
        if len(recientes) < 3:
            return 1.0, 1.0
        gf = float(np.mean([r[1] for r in recientes]))
        gc = float(np.mean([r[2] for r in recientes]))
        # Multiplicador suave: forma reciente ajusta el pronóstico, sin dominar toda la temporada.
        ataque = max(0.55, min(1.65, gf / max(0.1, (ph + pa) / 2)))
        defensa_permisiva = max(0.55, min(1.65, gc / max(0.1, (ph + pa) / 2)))
        return ataque, defensa_permisiva

    ataque_local, concede_local = factores_forma(local)
    ataque_visita, concede_visita = factores_forma(visita)
    ajuste_local = max(0.80, min(1.20, (ataque_local * concede_visita) ** 0.25))
    ajuste_visita = max(0.80, min(1.20, (ataque_visita * concede_local) ** 0.25))
    return max(0.15, lh * ajuste_local), max(0.15, av * ajuste_visita), ajuste_local, ajuste_visita


def resumen_forma(equipo, modelo, fecha_partido):
    formas = modelo[7]
    recientes = sorted(
        (r for r in formas.get(equipo, []) if r[0].astimezone(TZ) < fecha_partido),
        key=lambda r: r[0], reverse=True,
    )[:8]
    if not recientes:
        return "Sin resultados recientes disponibles"
    ganados = sum(gf > gc for _, gf, gc in recientes)
    empatados = sum(gf == gc for _, gf, gc in recientes)
    perdidos = len(recientes) - ganados - empatados
    gf_total = sum(gf for _, gf, _ in recientes)
    gc_total = sum(gc for _, _, gc in recientes)
    return f"{ganados}G–{empatados}E–{perdidos}P · goles {gf_total}:{gc_total} · {len(recientes)} partidos"


def matriz_corner(media):
    max_media = max(media)
    cola = nbinom.ppf(
        0.999999, DISPERSION_CORNERS,
        DISPERSION_CORNERS / (DISPERSION_CORNERS + max_media),
    )
    valores = np.arange(max(N_CORNERS, int(cola) + 1))
    p = np.outer(
        nbinom.pmf(valores, DISPERSION_CORNERS, DISPERSION_CORNERS / (DISPERSION_CORNERS + media[0])),
        nbinom.pmf(valores, DISPERSION_CORNERS, DISPERSION_CORNERS / (DISPERSION_CORNERS + media[1])),
    )
    return p / p.sum(), np.meshgrid(valores, valores, indexing="ij")


def construir_prediccion_partido(partido, modelo, corners):
    """Build the same pre-kickoff goal/corner matrices for the app and archive job."""
    local = partido["homeTeam"].get("shortName") or partido["homeTeam"]["name"]
    visita = partido["awayTeam"].get("shortName") or partido["awayTeam"]["name"]
    fecha_objetivo = datetime.fromisoformat(partido["utcDate"].replace("Z", "+00:00")).astimezone(TZ)
    estadisticas = calcular_estadisticas_goles(modelo[9], fecha_objetivo)
    modelo_objetivo = (modelo[0], *estadisticas, fecha_objetivo.date(), modelo[9])
    proporcion_primera = proporcion_goles_primer_tiempo(modelo[9], fecha_objetivo)
    gl, gv, ajuste_local, ajuste_visita = estimar_goles(local, visita, modelo_objetivo, fecha_objetivo)
    cola = int(poisson.ppf(0.999999, max(gl, gv))) + 1
    goles = np.arange(max(11, cola))
    i, j = np.meshgrid(goles, goles, indexing="ij")
    p_goles = np.outer(poisson.pmf(goles, gl), poisson.pmf(goles, gv))
    p_goles /= p_goles.sum()

    p_corners = ci = cj = None
    media_corners_local = media_corners_visita = None
    corners_final = None
    if partido.get("status") == "FINISHED" and corners is not None:
        corners_final = resultado_corners(corners, local, visita, fecha_objetivo)
    corners_objetivo = modelo_corners_antes_de_fecha(corners, fecha_objetivo) if corners is not None else None
    if corners_objetivo is not None:
        lk = clave_equipo(local, corners_objetivo["nombres"])
        vk = clave_equipo(visita, corners_objetivo["nombres"])
        if lk and vk:
            lh, va = corners_objetivo["equipos"][lk], corners_objetivo["equipos"][vk]
            media_l = media_corners_ponderada(lh["home_for"], corners_objetivo["prom_home"])
            contra_l = media_corners_ponderada(va["away_against"], corners_objetivo["prom_home"])
            media_v = media_corners_ponderada(va["away_for"], corners_objetivo["prom_away"])
            contra_v = media_corners_ponderada(lh["home_against"], corners_objetivo["prom_away"])
            media_corners_local = (media_l + contra_l) / 2
            media_corners_visita = (media_v + contra_v) / 2
            p_corners, (ci, cj) = matriz_corner((media_corners_local, media_corners_visita))

    resultado = None
    if partido.get("status") == "FINISHED":
        marcador = marcador_reglamentario(partido)
        if marcador.get("home") is not None and marcador.get("away") is not None:
            half_time = partido.get("score", {}).get("halfTime", {})
            primer_tiempo = segundo_tiempo = None
            if half_time.get("home") is not None and half_time.get("away") is not None:
                primer_tiempo = (int(half_time["home"]), int(half_time["away"]))
                segundo_tiempo = (
                    int(marcador["home"]) - primer_tiempo[0],
                    int(marcador["away"]) - primer_tiempo[1],
                )
            resultado = {
                "partido": (int(marcador["home"]), int(marcador["away"])),
                "primer_tiempo": primer_tiempo,
                "segundo_tiempo": segundo_tiempo,
            }
    return {
        "local": local, "visita": visita, "fecha": fecha_objetivo,
        "modelo": modelo_objetivo, "gl": gl, "gv": gv,
        "ajuste_local": ajuste_local, "ajuste_visita": ajuste_visita,
        "p_goles": p_goles, "i": i, "j": j,
        "p_corners": p_corners, "ci": ci, "cj": cj,
        "media_corners_local": media_corners_local,
        "media_corners_visita": media_corners_visita,
        "corners_final": corners_final, "resultado": resultado,
        "proporcion_primera": proporcion_primera,
    }


def leer_historial_fijo():
    """Read the append-only forecast and settlement ledgers, if they exist."""
    columnas_pred = [
        "prediction_id", "match_id", "fecha", "local", "visita", "categoria",
        "jugada", "probabilidad_base", "probabilidad_calibrada", "calibracion", "precio", "disponibilidad", "ticker",
        "clave_interna", "capturado_en",
    ]
    columnas_res = ["prediction_id", "estado", "marcador", "resuelto_en"]
    try:
        predicciones = pd.read_csv(PREDICCIONES_ARCHIVO, dtype={"prediction_id": str, "match_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        predicciones = pd.DataFrame(columns=columnas_pred)
    try:
        resultados = pd.read_csv(RESULTADOS_ARCHIVO, dtype={"prediction_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        resultados = pd.DataFrame(columns=columnas_res)
    return predicciones, resultados


def calibrar_resumen(resumen, predicciones, resultados):
    """Conservatively learn category calibration from prior, settled snapshots only."""
    if not resumen:
        return resumen
    observados = {}
    if not predicciones.empty and not resultados.empty:
        unidos = predicciones.merge(resultados, on="prediction_id", how="inner")
        unidos = unidos[unidos["estado"].isin(["WIN", "LOSS"])]
        if not unidos.empty:
            unidos = unidos.sort_values("capturado_en").drop_duplicates(["match_id", "categoria"], keep="last")
        for _, fila in unidos.iterrows():
            try:
                p = float(fila["probabilidad_base"])
            except (TypeError, ValueError):
                continue
            if 0 <= p <= 1:
                observados.setdefault(str(fila["categoria"]), []).append((p, fila["estado"] == "WIN"))

    salida = []
    for fila in resumen:
        copia = dict(fila)
        base = float(copia.get("Probabilidad base", copia.get("_p", 0.0)))
        datos = observados.get(str(copia.get("Tipo", "")), [])
        cercanos = [(p, acierto) for p, acierto in datos if abs(p - base) <= 0.10]
        ajustada = base
        # Avoid fitting a noisy category until a useful out-of-sample sample exists.
        if len(cercanos) >= 30:
            ganados = sum(acierto for _, acierto in cercanos)
            ajustada = (ganados + 20 * base) / (len(cercanos) + 20)
        copia["Probabilidad base"] = base
        copia["Probabilidad modelo"] = f"{ajustada:.1%}"
        copia["Calibración"] = f"Aprendida con {len(cercanos)} resultados" if len(cercanos) >= 30 else "Sin calibrar: faltan 30 casos comparables"
        salida.append(copia)
    return salida


def resumen_guardado(match_id, predicciones, resultados):
    """Return the immutable pre-match summary joined with its separately appended result."""
    if predicciones.empty:
        return []
    filas = predicciones[predicciones["match_id"].astype(str) == str(match_id)].copy()
    if filas.empty:
        return []
    estados = resultados.set_index("prediction_id")["estado"].to_dict() if not resultados.empty else {}
    capturas = []
    filas = filas.sort_values("capturado_en").drop_duplicates("categoria", keep="last")
    for _, fila in filas.sort_values(["categoria"]).iterrows():
        p = float(fila.get("probabilidad_calibrada", fila["probabilidad_base"]))
        capturado = datetime.fromisoformat(str(fila.get("capturado_en", ""))).astimezone(TZ).strftime("%d/%m %H:%M hora Chicago")
        capturas.append({
            "Tipo": fila["categoria"],
            "Jugada sencilla": fila["jugada"],
            "Probabilidad modelo": f"{p:.1%}",
            "Calibración": fila.get("calibracion", "Sin dato"),
            "Resultado": {"WIN": "✅ Se cumplió", "LOSS": "❌ No se cumplió", "UNKNOWN": "Sin dato",
                "NO_EVALUABLE": "⚠️ NO EVALUABLE · Falta dato oficial",
            }.get(estados.get(fila["prediction_id"]), "Pendiente de resultado"),
            "Precio ahora": fila.get("precio", "—"),
            "Disponibilidad": fila.get("disponibilidad", "—"),
            "Ticker": fila.get("ticker", "—"),
            "capturado_en": capturado,
        })
    return capturas


def historial_guardado(match_id, predicciones, resultados):
    """Show every immutable snapshot for a fixture, not only the latest card."""
    if predicciones.empty:
        return pd.DataFrame()
    filas = predicciones[predicciones["match_id"].astype(str) == str(match_id)].copy()
    if filas.empty:
        return pd.DataFrame()
    estados = resultados.set_index("prediction_id")["estado"].to_dict() if not resultados.empty else {}
    filas["Resultado"] = filas["prediction_id"].map(estados).map(
        {
            "WIN": "✅ WIN · Se cumplió",
            "LOSS": "❌ LOSS · No se cumplió",
            "UNKNOWN": "Sin dato",
            "NO_EVALUABLE": "⚠️ NO EVALUABLE · Falta dato oficial",
        }
    ).fillna("Pendiente de resultado")
    filas["Captura (Chicago)"] = pd.to_datetime(filas["capturado_en"], utc=True).dt.tz_convert(TZ).dt.strftime("%d/%m %H:%M")
    filas["Probabilidad base"] = pd.to_numeric(filas["probabilidad_base"], errors="coerce").map(lambda x: f"{x:.1%}")
    filas["Probabilidad guardada"] = pd.to_numeric(filas["probabilidad_calibrada"], errors="coerce").map(lambda x: f"{x:.1%}")
    return filas.rename(columns={
        "categoria": "Tipo", "jugada": "Pronóstico", "precio": "Precio Kalshi",
        "disponibilidad": "Disponibilidad", "ticker": "Ticker",
    })[["Captura (Chicago)", "Tipo", "Pronóstico", "Probabilidad base", "Probabilidad guardada", "Resultado", "Precio Kalshi", "Disponibilidad", "Ticker"]].sort_values("Captura (Chicago)", ascending=False)


def metricas_historial(predicciones, resultados):
    """Out-of-sample scorecard for immutable snapshots with known outcomes."""
    if predicciones.empty or resultados.empty:
        return pd.DataFrame(), 0
    unidos = predicciones.merge(resultados, on="prediction_id", how="inner")
    unidos = unidos[unidos["estado"].isin(["WIN", "LOSS"])].copy()
    if unidos.empty:
        return pd.DataFrame(), 0
    unidos = unidos.sort_values("capturado_en").drop_duplicates(["match_id", "categoria"], keep="last")
    unidos["p"] = pd.to_numeric(unidos["probabilidad_calibrada"], errors="coerce")
    unidos = unidos[unidos["p"].between(0, 1)]
    unidos["acierto"] = (unidos["estado"] == "WIN").astype(float)
    unidos["Brier"] = (unidos["p"] - unidos["acierto"]) ** 2
    tabla = unidos.groupby("categoria").agg(
        Pronosticos=("prediction_id", "count"),
        Aciertos=("acierto", "sum"),
        tasa=("acierto", "mean"),
        brier=("Brier", "mean"),
    ).reset_index().rename(columns={"categoria": "Mercado", "tasa": "Tasa de acierto", "brier": "Brier medio"})
    tabla["Tasa de acierto"] = tabla["Tasa de acierto"].map(lambda x: f"{x:.1%}")
    tabla["Brier medio"] = tabla["Brier medio"].map(lambda x: f"{x:.3f}")
    return tabla.sort_values("Pronosticos", ascending=False), len(unidos)
