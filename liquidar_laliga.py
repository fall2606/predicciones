"""Liquidación automática e independiente de pronósticos de LaLiga ya archivados.

No recalcula ni modifica predicciones; sólo añade filas a resultados_laliga.csv
cuando la fuente oficial confirma que el partido terminó y el resultado es evaluable.
"""
import ast
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from laliga_engine import (
    TZ,
    PREDICCIONES_ARCHIVO,
    RESULTADOS_ARCHIVO,
    cargar_corners,
    evaluar_yes_goles,
    resultado_del_periodo,
    resultado_corners,
)

API = "https://api.football-data.org/v4"
COLUMNAS_RESULTADO = ["prediction_id", "estado", "marcador", "resuelto_en"]


def cargar_partidos(clave, temporadas):
    """Descarga fixtures de temporada actual y anterior; sigue aunque una falle."""
    partidos = {}
    for temporada in temporadas:
        try:
            response = requests.get(
                f"{API}/competitions/PD/matches",
                headers={"X-Auth-Token": clave},
                params={"season": temporada},
                timeout=30,
            )
            response.raise_for_status()
            for partido in response.json().get("matches", []):
                partidos[str(partido.get("id"))] = partido
        except requests.RequestException as exc:
            print(f"Aviso: no pude consultar LaLiga {temporada}: {exc}")
    return partidos


def leer_csv(path, columnas, dtype=None):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columnas)
    try:
        return pd.read_csv(path, dtype=dtype or str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columnas)


def marcador_periodos(partido):
    score = partido.get("score") or {}
    full = score.get("fullTime") or {}
    if full.get("home") is None or full.get("away") is None:
        return None
    local, visita = int(full["home"]), int(full["away"])
    half = score.get("halfTime") or {}
    primer_tiempo = None
    segundo_tiempo = None
    if half.get("home") is not None and half.get("away") is not None:
        h, a = int(half["home"]), int(half["away"])
        primer_tiempo = (h, a)
        segundo_tiempo = (local - h, visita - a)
    return {
        "partido": (local, visita),
        "primer_tiempo": primer_tiempo,
        "segundo_tiempo": segundo_tiempo,
    }


def primer_anotador(partido, marcador):
    goles = [
        g for g in (partido.get("goals") or [])
        if isinstance(g, dict) and isinstance(g.get("team"), dict)
    ]

    def orden(g):
        try:
            minuto = int(g.get("minute") or 0)
        except (TypeError, ValueError):
            minuto = 0
        try:
            añadido = int(g.get("injuryTime") or 0)
        except (TypeError, ValueError):
            añadido = 0
        return minuto, añadido

    goles.sort(key=orden)
    if goles:
        equipo_id = goles[0]["team"].get("id")
        if equipo_id == (partido.get("homeTeam") or {}).get("id"):
            return "local"
        if equipo_id == (partido.get("awayTeam") or {}).get("id"):
            return "visita"
        return None
    if sum(marcador["partido"]) == 0:
        return "sin_goles"
    # Hubo goles pero el proveedor aún no ofrece los eventos: reintentar más tarde.
    return None


def evaluar_pronostico(fila, partido, marcador, corners):
    try:
        clave, lado = ast.literal_eval(str(fila.get("clave_interna", "")))
        serie = clave[0]
    except (ValueError, SyntaxError, TypeError, IndexError):
        return None

    if serie in {"KXLALIGAFTTS", "KXLALIGAFIRSTGOAL"}:
        primer_gol = primer_anotador(partido, marcador)
        if primer_gol is None:
            return None
        jugada = str(fila.get("jugada", "")).strip()
        local = str(fila.get("local", ""))
        visita = str(fila.get("visita", ""))
        if jugada == f"Primer gol de {local}" or jugada == f"Primer gol de {local} o sin goles":
            acierto = primer_gol in ({"local", "sin_goles"} if "o sin goles" in jugada else {"local"})
        elif jugada == f"Primer gol de {visita}" or jugada == f"Primer gol de {visita} o sin goles":
            acierto = primer_gol in ({"visita", "sin_goles"} if "o sin goles" in jugada else {"visita"})
        elif jugada == "No habrá goles":
            acierto = primer_gol == "sin_goles"
        elif jugada == "Habrá al menos un gol":
            acierto = primer_gol != "sin_goles"
        else:
            acierto_yes = primer_gol == clave[1]
            acierto = acierto_yes if lado == "YES" else not acierto_yes
        return "WIN" if acierto else "LOSS"

    if serie in {"KXLALIGACORNERS", "KXLALIGATCORNERS"}:
        if corners is None:
            return None
        try:
            fecha = datetime.fromisoformat(str(fila["fecha"]).replace("Z", "+00:00"))
            corners_final = resultado_corners(corners, fila["local"], fila["visita"], fecha)
        except (KeyError, TypeError, ValueError):
            return None
        if corners_final is None:
            return None
        cuenta = (
            sum(corners_final)
            if serie == "KXLALIGACORNERS"
            else corners_final[0] if clave[1] == "local" else corners_final[1]
        )
        linea, direccion = clave[2], clave[3]
        acierto_yes = cuenta < linea if direccion == "under" else cuenta >= linea
        acierto = acierto_yes if lado == "YES" else not acierto_yes
        return "WIN" if acierto else "LOSS"

    marcador_periodo = resultado_del_periodo(clave, marcador)
    acierto_yes = evaluar_yes_goles(clave, marcador_periodo)
    if acierto_yes is None:
        return None
    acierto = acierto_yes if lado == "YES" else not acierto_yes
    return "WIN" if acierto else "LOSS"


def main():
    clave = os.environ["CLAVE"]
    ahora = datetime.now(TZ)
    temporada = ahora.year if ahora.month >= 7 else ahora.year - 1
    temporadas = [temporada, temporada - 1]
    predicciones = leer_csv(
        PREDICCIONES_ARCHIVO,
        ["prediction_id", "match_id", "fecha", "local", "visita", "jugada", "clave_interna"],
        {"prediction_id": str, "match_id": str},
    )
    resultados = leer_csv(
        RESULTADOS_ARCHIVO,
        COLUMNAS_RESULTADO,
        {"prediction_id": str},
    )
    if predicciones.empty:
        print("No hay pronósticos guardados para liquidar.")
        return

    resueltos = set(resultados["prediction_id"].astype(str)) if not resultados.empty else set()
    pendientes = predicciones[~predicciones["prediction_id"].astype(str).isin(resueltos)]
    if pendientes.empty:
        print("No quedan pronósticos pendientes.")
        return

    partidos = cargar_partidos(clave, temporadas)
    codigos_corners = [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in temporadas]
    try:
        corners = cargar_corners(codigos_corners)
    except Exception as exc:
        print(f"Aviso: corners no disponibles en esta ejecución: {exc}")
        corners = None

    nuevos = []
    partidos_vistos = set()
    for _, fila in pendientes.iterrows():
        match_id = str(fila.get("match_id", ""))
        partido = partidos.get(match_id)
        if not partido or partido.get("status") not in {"FINISHED", "AWARDED"}:
            continue
        marcador = marcador_periodos(partido)
        if marcador is None:
            continue
        estado = evaluar_pronostico(fila, partido, marcador, corners)
        if estado not in {"WIN", "LOSS"}:
            continue
        goles = marcador["partido"]
        nuevos.append({
            "prediction_id": str(fila["prediction_id"]),
            "estado": estado,
            "marcador": f"{goles[0]}-{goles[1]}",
            "resuelto_en": datetime.now().astimezone().isoformat(),
        })
        resueltos.add(str(fila["prediction_id"]))
        partidos_vistos.add(match_id)

    if nuevos:
        nuevo_archivo = not RESULTADOS_ARCHIVO.exists() or RESULTADOS_ARCHIVO.stat().st_size == 0
        with RESULTADOS_ARCHIVO.open("a", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_RESULTADO)
            if nuevo_archivo:
                escritor.writeheader()
            escritor.writerows(nuevos)
    print(
        f"Partidos finalizados revisados: {len(partidos_vistos)} | "
        f"Pronósticos liquidados: {len(nuevos)} | "
        f"Siguen pendientes por falta de resultado oficial/datos: {len(pendientes) - len(nuevos)}"
    )


if __name__ == "__main__":
    main()
