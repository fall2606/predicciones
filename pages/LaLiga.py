from datetime import datetime, timedelta
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor, as_completed
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
st.title("🇪🇸 LaLiga · mercados y probabilidades de Kalshi")
st.caption(
    "Partidos próximos de LaLiga. Revisa todos los tipos y líneas que Kalshi ofrece en la liga, "
    "con probabilidad estimada por selección YES y NO."
)
st.link_button("Abrir Kalshi", "https://kalshi.com/")


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
    """Tipo y línea observados en un contrato real de LaLiga, sin depender de equipos."""
    serie = evento.get("series_ticker", "")
    series_con_linea = {
        "KXLALIGASPREAD", "KXLALIGATOTAL", "KXLALIGACORNERS",
        "KXLALIGATCORNERS", "KXLALIGATEAMTOTAL", "KXLALIGA1HSPREAD",
        "KXLALIGA1HTOTAL", "KXLALIGA2HTOTAL",
    }
    series_sin_linea = {
        "KXLALIGAGAME", "KXLALIGABTTS", "KXLALIGAFTTS", "KXLALIGAFIRSTGOAL",
        "KXLALIGA1H", "KXLALIGA1HBTTS", "KXLALIGA2H",
    }
    if serie in series_sin_linea:
        return serie, None, None
    linea = _linea_mercado(mercado)
    if serie in series_con_linea and linea is not None:
        direccion = _direccion_mercado(mercado) if serie in {
            "KXLALIGATOTAL", "KXLALIGACORNERS", "KXLALIGATCORNERS", "KXLALIGATEAMTOTAL",
            "KXLALIGA1HTOTAL", "KXLALIGA2HTOTAL",
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
    if serie in {"KXLALIGAGAME", "KXLALIGA1H", "KXLALIGA2H"}:
        lado = limpiar(mercado.get("yes_sub_title") or mercado.get("title", ""))
        if "tie" in lado or "draw" in lado or lado == "empate":
            return serie, "empate", None, None
        equipo = buscar_equipo_en_texto(lado, local, visita)
        return (serie, "local" if equipo == local else "visita", None, None) if equipo else None
    if serie in {"KXLALIGASPREAD", "KXLALIGA1HSPREAD"}:
        equipo = buscar_equipo_en_texto(texto, local, visita)
        return (serie, "local" if equipo == local else "visita", linea, direccion) if equipo else None
    if serie in {"KXLALIGATOTAL", "KXLALIGACORNERS", "KXLALIGA1HTOTAL", "KXLALIGA2HTOTAL"}:
        return serie, "over", linea, direccion
    if serie in {"KXLALIGATCORNERS", "KXLALIGATEAMTOTAL"}:
        equipo = buscar_equipo_en_texto(texto, local, visita)
        return (serie, "local" if equipo == local else "visita", linea, direccion) if equipo else None
    if serie in {"KXLALIGABTTS", "KXLALIGA1HBTTS"}:
        return serie, "ambos", None, None
    if serie in {"KXLALIGAFTTS", "KXLALIGAFIRSTGOAL"}:
        if "no goal" in texto or "sin gol" in texto:
            return serie, "sin_goles", None, None
        equipo = buscar_equipo_en_texto(texto, local, visita)
        return (serie, "local" if equipo == local else "visita", None, None) if equipo else None
    return None


def modelar_seleccion_kalshi(clave, local, visita, p_goles, i, j, p_corners, ci, cj):
    serie, lado, linea, direccion = clave
    dominio, condicion = "goles", None
    periodo = 0.45 if serie.startswith("KXLALIGA1H") else 0.55 if serie.startswith("KXLALIGA2H") else None
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
    if serie in {"KXLALIGAGAME", "KXLALIGA1H", "KXLALIGA2H"}:
        condicion = i == j if lado == "empate" else i > j if lado == "local" else j > i
    elif serie in {"KXLALIGASPREAD", "KXLALIGA1HSPREAD"} and linea is not None:
        margen = i - j if lado == "local" else j - i
        condicion = margen > linea
    elif serie in {"KXLALIGATOTAL", "KXLALIGA1HTOTAL", "KXLALIGA2HTOTAL"} and linea is not None:
        condicion = i + j < linea if direccion == "under" else i + j > linea
    elif serie == "KXLALIGACORNERS" and p_corners is not None and ci is not None and cj is not None and linea is not None:
        dominio, condicion = "corners", ci + cj < linea if direccion == "under" else ci + cj >= linea
    elif serie == "KXLALIGATCORNERS" and p_corners is not None and ci is not None and cj is not None and linea is not None:
        dominio, condicion = "corners", (ci if lado == "local" else cj) < linea if direccion == "under" else (ci if lado == "local" else cj) >= linea
    elif serie == "KXLALIGATEAMTOTAL" and linea is not None:
        condicion = (i if lado == "local" else j) < linea if direccion == "under" else (i if lado == "local" else j) > linea
    elif serie in {"KXLALIGABTTS", "KXLALIGA1HBTTS"}:
        condicion = (i > 0) & (j > 0)
    elif serie in {"KXLALIGAFTTS", "KXLALIGAFIRSTGOAL"}:
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


def etiqueta_seleccion_kalshi(clave, local, visita):
    serie, lado, linea, direccion = clave
    equipo = local if lado == "local" else visita
    periodo = " (1.ª parte)" if serie.startswith("KXLALIGA1H") else " (2.ª parte)" if serie.startswith("KXLALIGA2H") else ""
    if serie in {"KXLALIGAGAME", "KXLALIGA1H", "KXLALIGA2H"}:
        return ("Empate" if lado == "empate" else f"Gana {equipo}") + periodo
    if serie in {"KXLALIGASPREAD", "KXLALIGA1HSPREAD"}:
        return f"{equipo} gana por más de {linea:g} goles{periodo}"
    if serie in {"KXLALIGATOTAL", "KXLALIGA1HTOTAL", "KXLALIGA2HTOTAL"}:
        return (f"Menos de {linea:g} goles totales{periodo}" if direccion == "under"
                else f"Más de {linea:g} goles totales{periodo}")
    if serie == "KXLALIGACORNERS":
        return f"Menos de {linea:g} corners totales" if direccion == "under" else f"{linea:g}+ corners totales"
    if serie == "KXLALIGATCORNERS":
        return f"Menos de {linea:g} corners de {equipo}" if direccion == "under" else f"{linea:g}+ corners de {equipo}"
    if serie == "KXLALIGATEAMTOTAL":
        return f"Menos de {linea:g} goles de {equipo}" if direccion == "under" else f"Más de {linea:g} goles de {equipo}"
    if serie in {"KXLALIGABTTS", "KXLALIGA1HBTTS"}:
        return "Ambos equipos marcan" + periodo
    if lado == "sin_goles":
        return "No se marca ningún gol"
    return f"Primer gol de {equipo}"


def etiqueta_no_kalshi(clave, local, visita):
    serie, lado, linea, direccion = clave
    periodo = " (1.ª parte)" if serie.startswith("KXLALIGA1H") else " (2.ª parte)" if serie.startswith("KXLALIGA2H") else ""
    if serie in {"KXLALIGAGAME", "KXLALIGA1H", "KXLALIGA2H"}:
        otro = visita if lado == "local" else local
        return f"No gana {local if lado == 'local' else visita} (empate o gana {otro})" if lado != "empate" else "No hay empate (gana local o visita)"
    if serie in {"KXLALIGATOTAL", "KXLALIGA1HTOTAL", "KXLALIGA2HTOTAL"}:
        opuesta = "Más de" if direccion == "under" else "Menos de"
        return f"NO: {opuesta} {linea:g} goles totales{periodo}"
    if serie == "KXLALIGACORNERS":
        opuesta = f"{linea:g}+" if direccion == "under" else f"Menos de {linea:g}"
        return f"NO: {opuesta} corners totales"
    if serie == "KXLALIGATCORNERS":
        opuesta = f"{linea:g}+" if direccion == "under" else f"Menos de {linea:g}"
        return f"NO: {opuesta} corners de {local if lado == 'local' else visita}"
    if serie == "KXLALIGATEAMTOTAL":
        opuesta = "Más de" if direccion == "under" else "Menos de"
        return f"NO: {opuesta} {linea:g} goles de {local if lado == 'local' else visita}{periodo}"
    if serie in {"KXLALIGABTTS", "KXLALIGA1HBTTS"}:
        return "Al menos un equipo no marca" + periodo
    if serie in {"KXLALIGAFTTS", "KXLALIGAFIRSTGOAL"}:
        return "El primer gol no es de " + ("nadie" if lado == "sin_goles" else local if lado == "local" else visita)
    return f"NO: {etiqueta_seleccion_kalshi(clave, local, visita)}"


def preparar_top_predicciones_kalshi(eventos, local, visita, fecha, p_goles, i, j,
                                    p_corners, ci, cj, resultado=None, corners_final=None):
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
                    modelar_seleccion_kalshi(clave, local, visita, p_goles, i, j, p_corners, ci, cj)
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
        if serie in {"KXLALIGAGAME", "KXLALIGA1H", "KXLALIGA2H"}:
            claves.update({(serie, "local", None, None), (serie, "empate", None, None), (serie, "visita", None, None)})
        elif serie in {"KXLALIGASPREAD", "KXLALIGATCORNERS", "KXLALIGATEAMTOTAL", "KXLALIGA1HSPREAD"}:
            claves.update({(serie, "local", linea, direccion), (serie, "visita", linea, direccion)})
        elif serie in {"KXLALIGAFTTS", "KXLALIGAFIRSTGOAL"}:
            claves.update({(serie, "local", None, None), (serie, "visita", None, None), (serie, "sin_goles", None, None)})
        elif serie in {"KXLALIGABTTS", "KXLALIGA1HBTTS"}:
            claves.add((serie, "ambos", None, None))
        else:
            claves.add((serie, "over", linea, direccion))

    predicciones = []
    for clave in claves:
        estimacion = modelar_seleccion_kalshi(clave, local, visita, p_goles, i, j, p_corners, ci, cj)
        if estimacion is None:
            continue
        dominio, condicion_yes, probabilidad_yes = estimacion
        mercado = ofertas_actuales.get(clave)
        for lado_apuesta in ("YES", "NO"):
            probabilidad = probabilidad_yes if lado_apuesta == "YES" else 1 - probabilidad_yes
            condicion = condicion_yes if lado_apuesta == "YES" else 1 - np.asarray(condicion_yes, dtype=float)
            precio_campo = "yes_ask_dollars" if lado_apuesta == "YES" else "no_ask_dollars"
            precio = cotizacion_valida(mercado.get(precio_campo)) if mercado else None
            estado = (
                "Disponible ahora" if mercado and precio is not None else
                f"Kalshi lista el contrato, sin oferta {lado_apuesta}" if mercado else
                "Tipo/línea ofrecido en LaLiga; falta abrirlo para este partido"
            )

            resultado_texto = "Pendiente"
            if resultado is not None and dominio == "goles":
                if clave[0] in {"KXLALIGAFTTS", "KXLALIGAFIRSTGOAL"} and all(g > 0 for g in resultado):
                    resultado_texto = "Sin dato de primer anotador"
                else:
                    acierto_yes = bool(condicion_yes[resultado[0], resultado[1]])
                    acierto = acierto_yes if lado_apuesta == "YES" else not acierto_yes
                    resultado_texto = "✅ Se cumplió" if acierto else "❌ No se cumplió"
            elif corners_final is not None and dominio == "corners":
                acierto_yes = bool(condicion_yes[int(corners_final[0]), int(corners_final[1])])
                acierto = acierto_yes if lado_apuesta == "YES" else not acierto_yes
                resultado_texto = "✅ Se cumplió" if acierto else "❌ No se cumplió"

            predicciones.append({
                "Apuesta": etiqueta_seleccion_kalshi(clave, local, visita) if lado_apuesta == "YES" else etiqueta_no_kalshi(clave, local, visita),
                "Lado Kalshi": lado_apuesta,
                "Probabilidad del modelo": f"{probabilidad:.1%}",
                "Precio de compra": f"{precio:.0%}" if precio is not None else "—",
                "Disponibilidad": estado,
                "Ticker": mercado.get("ticker", "") if mercado else "—",
                "Resultado": resultado_texto,
                "_p": probabilidad,
            })

    predicciones.sort(key=lambda x: x["_p"], reverse=True)
    todas = [{k: v for k, v in fila.items() if k != "_p"} for fila in predicciones]
    top = todas[:10]
    mercados_actuales.sort(key=lambda x: x["Mercado Kalshi"])
    return top, todas, mercados_actuales, len(plantillas)


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
    if inicio <= fecha_local <= fin and estado in {"SCHEDULED", "TIMED"}:
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
    etiqueta = f"{fecha:%a %d/%m %H:%M} · {local} vs {visita}"
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
media_corners_local = media_corners_visita = None
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
        media_corners_local = (media_l + contra_l) / 2
        media_corners_visita = (media_v + contra_v) / 2
        p_corners, (ci, cj) = matriz_corner((media_corners_local, media_corners_visita))
        corners_final = corners["resultados"].get((limpiar(local), limpiar(visita)))

resultado = None
if partido["status"] == "FINISHED":
    marcador = partido.get("score", {}).get("fullTime", {})
    if marcador.get("home") is not None and marcador.get("away") is not None:
        resultado = (int(marcador["home"]), int(marcador["away"]))

try:
    eventos_kalshi, series_kalshi_con_error, total_series_kalshi = cargar_eventos_kalshi_laliga()
    fecha_partido = datetime.fromisoformat(partido["utcDate"].replace("Z", "+00:00")).astimezone(TZ).date()
    top_kalshi, todas_kalshi, mercados_kalshi, num_plantillas_kalshi = preparar_top_predicciones_kalshi(
        eventos_kalshi, local, visita, fecha_partido, p_goles, i, j, p_corners, ci, cj, resultado, corners_final
    )
except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
    eventos_kalshi, series_kalshi_con_error, total_series_kalshi = [], [], 0
    top_kalshi, todas_kalshi, mercados_kalshi, num_plantillas_kalshi = [], [], [], 0
    st.error(f"No pude consultar los mercados abiertos de Kalshi: {exc}")

st.subheader(f"{local} vs {visita}")
st.markdown("### Las 10 apuestas individuales más probables")
st.caption(
    "Incluye YES y NO de cada contrato observado entre los mercados de LaLiga de Kalshi, con sus precios cuando ya está abierto para este partido. "
    "El NO es el complemento exacto del YES (por ejemplo, NO a ‘gana el local’ = empate o victoria visitante). No se combinan."
)
if top_kalshi:
    st.dataframe(pd.DataFrame(top_kalshi), use_container_width=True, hide_index=True)
else:
    st.info("No encontré tipos y líneas activos de Kalshi en LaLiga con estimación suficiente para este partido.")
st.write(f"**Marcador esperado:** {gl:.1f}–{gv:.1f} goles")

with st.expander("Cómo calcula el algoritmo estas probabilidades"):
    st.markdown(
        "**Goles:** estima los goles esperados de cada equipo con promedios de LaLiga y fuerzas de ataque/defensa "
        "local y visitante suavizadas con el promedio de liga (K=6). Luego calcula una distribución Poisson para cada equipo "
        "y forma la matriz de marcadores. La probabilidad de cada YES es la suma de las celdas que cumplen esa condición; "
        "la del NO es 1 menos la del YES."
    )
    st.write(f"λ estimada: {local} {gl:.2f} goles · {visita} {gv:.2f} goles")
    st.markdown(
        "**Corners:** estima corners por equipo con registros locales/visitantes de football-data.co.uk, "
        "suavizados hacia la media de liga (K=6), y una distribución binomial negativa (dispersión=16). "
        "Las probabilidades de corners se calculan sumando los resultados de esa matriz; el NO también es el complemento del YES."
    )
    if media_corners_local is not None and media_corners_visita is not None:
        st.write(f"Corners esperados: {local} {media_corners_local:.2f} · {visita} {media_corners_visita:.2f}")
    else:
        st.info("No hay datos suficientes de corners para calcular esas probabilidades.")
    st.markdown(
        "**Primer y segundo tiempo:** como la fuente no ofrece estadísticas específicas del descanso, "
        "el modelo reparte provisionalmente los goles esperados en 45% para la primera parte y 55% para la segunda. "
        "Estas probabilidades son más aproximadas que las de partido completo."
    )
    st.caption("Son probabilidades estimadas por un modelo estadístico; todavía no se calibran mediante un backtest de aciertos.")

if todas_kalshi:
    with st.expander(f"Ver todas las apuestas y líneas detectadas ({len(todas_kalshi)})"):
        st.dataframe(pd.DataFrame(todas_kalshi), use_container_width=True, hide_index=True)

with st.expander("Ver contratos abiertos ahora para este partido"):
    if mercados_kalshi:
        st.dataframe(pd.DataFrame(mercados_kalshi), use_container_width=True, hide_index=True)
    elif total_series_kalshi and not series_kalshi_con_error:
        st.info("Kalshi no tiene contratos abiertos para este encuentro y fecha.")
    else:
        st.info("No pude confirmar los contratos abiertos para este encuentro.")

if series_kalshi_con_error:
    st.warning(f"Kalshi no respondió para {len(series_kalshi_con_error)} de {total_series_kalshi} series de LaLiga; el listado puede estar incompleto.")

st.caption(
    "Las filas ‘Tipo/línea ofrecido en LaLiga; falta abrirlo para este partido’ usan una línea real detectada en otros partidos de la liga. "
    "Sirven como posibilidades del modelo; Kalshi podría no abrir ese contrato para este encuentro."
)
st.warning(
    "Las probabilidades son estimaciones y no garantías. Para apostar en Kalshi, verifica que el contrato exacto esté disponible "
    "para este partido y revisa las reglas del mercado."
)
