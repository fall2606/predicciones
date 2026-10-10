"""Append immutable LaLiga forecast snapshots and a separate settlement ledger."""

import ast
import csv
import os
from datetime import datetime, timedelta, timezone
import requests

from laliga_engine import (
    TZ,
    PREDICCIONES_ARCHIVO,
    RESULTADOS_ARCHIVO,
    cargar_corners,
    cargar_eventos_kalshi_laliga,
    cargar_partidos_y_estadisticas,
    calibrar_resumen,
    construir_prediccion_partido,
    evaluar_yes_goles,
    preparar_top_predicciones_kalshi,
    resultado_del_periodo,
    resultado_corners,
    leer_historial_fijo,
)

COLUMNAS_PRED = [
    "prediction_id", "match_id", "fecha", "local", "visita", "categoria", "jugada",
    "probabilidad_base", "probabilidad_calibrada", "calibracion", "precio", "disponibilidad", "ticker",
    "clave_interna", "capturado_en",
]
COLUMNAS_RESULTADO = ["prediction_id", "estado", "marcador", "resuelto_en"]


def anexar_csv(path, columnas, filas):
    nuevo = not path.exists() or path.stat().st_size == 0
    if not filas and not nuevo:
        return
    with path.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, extrasaction="ignore")
        if nuevo:
            escritor.writeheader()
        escritor.writerows(filas)


def main():
    clave = os.environ["CLAVE"]
    ahora = datetime.now(TZ)
    temporada = ahora.year if ahora.month >= 7 else ahora.year - 1
    anterior = temporada - 1
    codigos_corners = [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in (anterior, temporada)]

    modelo = cargar_partidos_y_estadisticas(clave, temporada, anterior)
    corners = cargar_corners(codigos_corners)
    predicciones, resultados = leer_historial_fijo()
    anexar_csv(PREDICCIONES_ARCHIVO, COLUMNAS_PRED, [])
    anexar_csv(RESULTADOS_ARCHIVO, COLUMNAS_RESULTADO, [])
    try:
        eventos, errores_series, total_series = cargar_eventos_kalshi_laliga()
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        eventos, errores_series, total_series = [], [f"No se pudo listar LaLiga: {exc}"], 0
    ids_existentes = set(predicciones["prediction_id"].astype(str)) if not predicciones.empty else set()
    ids_resueltos = set(resultados["prediction_id"].astype(str)) if not resultados.empty else set()

    nuevas = []
    captura = datetime.now(timezone.utc)
    sufijo_captura = captura.strftime("%Y%m%d%H")
    limite = datetime.combine(ahora.date() + timedelta(days=2), datetime.max.time(), tzinfo=TZ)
    for partido in modelo[0]:
        if partido.get("status") not in {"SCHEDULED", "TIMED"}:
            continue
        inicio = datetime.fromisoformat(partido["utcDate"].replace("Z", "+00:00"))
        if not (ahora < inicio.astimezone(TZ) <= limite):
            continue
        datos = construir_prediccion_partido(partido, modelo, corners)
        fecha = datos["fecha"].date()
        resultado_actual = preparar_top_predicciones_kalshi(
            eventos, datos["local"], datos["visita"], fecha,
            datos["p_goles"], datos["i"], datos["j"],
            datos["p_corners"], datos["ci"], datos["cj"], None, None,
            datos["proporcion_primera"],
        )[0]
        resultado_actual = calibrar_resumen(resultado_actual, predicciones, resultados)
        match_id = str(partido["id"])
        for fila in resultado_actual:
            prediction_id = f"{match_id}|{fila['Tipo']}|{sufijo_captura}"
            if prediction_id in ids_existentes:
                continue
            market_key, lado = ast.literal_eval(fila["Clave interna"])
            nuevas.append({
                "prediction_id": prediction_id,
                "match_id": match_id,
                "fecha": inicio.astimezone(TZ).isoformat(),
                "local": datos["local"],
                "visita": datos["visita"],
                "categoria": fila["Tipo"],
                "jugada": fila["Jugada sencilla"],
                "probabilidad_base": f"{float(fila['Probabilidad base']):.6f}",
                "probabilidad_calibrada": f"{float(fila['Probabilidad modelo'].rstrip('%')) / 100:.6f}",
                "calibracion": fila["Calibración"],
                "precio": fila["Precio ahora"],
                "disponibilidad": fila["Disponibilidad"],
                "ticker": fila.get("Ticker", "—"),
                "clave_interna": repr((market_key, lado)),
                "capturado_en": captura.isoformat(),
            })
            ids_existentes.add(prediction_id)

    anexar_csv(PREDICCIONES_ARCHIVO, COLUMNAS_PRED, nuevas)
    if nuevas:
        predicciones, resultados = leer_historial_fijo()
        ids_resueltos = set(resultados["prediction_id"].astype(str)) if not resultados.empty else set()

    # La API de la temporada actual puede omitir partidos ya cerrados o devolverlos
    # con estados desfasados. El historial validado (modelo[9]) contiene los partidos
    # finalizados con marcador; combinamos ambos por ID y damos prioridad a la respuesta
    # actual cuando esté disponible.
    # Partimos de la respuesta actual, pero el historial validado de partidos
    # finalizados debe prevalecer: la API de temporada puede conservar un estado
    # desfasado para un partido que ya tiene marcador final confirmado.
    partidos_por_id = {str(m["id"]): m for m in modelo[0]}
    partidos_por_id.update({str(m["id"]): m for m in modelo[9]})
    cierres = []
    for _, fila in predicciones.iterrows():
        prediction_id = str(fila["prediction_id"])
        if prediction_id in ids_resueltos:
            continue
        partido = partidos_por_id.get(str(fila["match_id"]))
        if not partido or partido.get("status") != "FINISHED":
            continue
        datos = construir_prediccion_partido(partido, modelo, corners)
        if not datos["resultado"]:
            continue
        try:
            market_key, lado = ast.literal_eval(fila["clave_interna"])
        except (ValueError, SyntaxError):
            continue
        serie = market_key[0]
        if serie in {"KXLALIGAFTTS", "KXLALIGAFIRSTGOAL"}:
            # En los mercados de primer gol, el marcador final por sí solo no
            # identifica al primer anotador. Usamos los goles del partido y la
            # selección guardada (jugada), que es la que ve el usuario.
            goles = partido.get("goals") or []
            goles_validos = [
                g for g in goles
                if isinstance(g, dict) and isinstance(g.get("team"), dict)
            ]
            goles_validos.sort(key=lambda g: (
                int((g.get("minute") or 0)),
                int((g.get("injuryTime") or 0)),
            ))
            total_goles = sum(datos["resultado"]["partido"])
            if goles_validos:
                primer_equipo_id = goles_validos[0]["team"].get("id")
                if primer_equipo_id == partido.get("homeTeam", {}).get("id"):
                    primer_gol = "local"
                elif primer_equipo_id == partido.get("awayTeam", {}).get("id"):
                    primer_gol = "visita"
                else:
                    continue
            elif total_goles == 0:
                primer_gol = "sin_goles"
            else:
                # Hay goles, pero el proveedor no incluyó el detalle necesario.
                continue

            jugada = str(fila.get("jugada", "")).strip()
            local = datos["local"]
            visita = datos["visita"]
            if jugada == f"Primer gol de {local}" or jugada == f"Primer gol de {local} o sin goles":
                acierto = primer_gol in ({"local", "sin_goles"} if "o sin goles" in jugada else {"local"})
            elif jugada == f"Primer gol de {visita}" or jugada == f"Primer gol de {visita} o sin goles":
                acierto = primer_gol in ({"visita", "sin_goles"} if "o sin goles" in jugada else {"visita"})
            elif jugada == "No habrá goles":
                acierto = primer_gol == "sin_goles"
            elif jugada == "Habrá al menos un gol":
                acierto = primer_gol != "sin_goles"
            else:
                # Compatibilidad para filas antiguas con una etiqueta inesperada.
                acierto_yes = primer_gol == market_key[1]
                acierto = acierto_yes if lado == "YES" else not acierto_yes
            estado = "WIN" if acierto else "LOSS"
        elif serie in {"KXLALIGACORNERS", "KXLALIGATCORNERS"}:
            if datos["corners_final"] is None:
                continue
            cuenta = (
                sum(datos["corners_final"])
                if serie == "KXLALIGACORNERS"
                else datos["corners_final"][0] if market_key[1] == "local"
                else datos["corners_final"][1]
            )
            linea, direccion = market_key[2], market_key[3]
            acierto_yes = cuenta < linea if direccion == "under" else cuenta >= linea
            acierto = acierto_yes if lado == "YES" else not acierto_yes
            estado = "WIN" if acierto else "LOSS"
        else:
            marcador = resultado_del_periodo(market_key, datos["resultado"])
            acierto_yes = evaluar_yes_goles(market_key, marcador)
            if acierto_yes is None:
                continue
            acierto = acierto_yes if lado == "YES" else not acierto_yes
            estado = "WIN" if acierto else "LOSS"
        marcador = datos["resultado"]["partido"]
        cierres.append({
            "prediction_id": prediction_id,
            "estado": estado,
            "marcador": f"{marcador[0]}-{marcador[1]}",
            "resuelto_en": datetime.now(timezone.utc).isoformat(),
        })
        ids_resueltos.add(prediction_id)

    anexar_csv(RESULTADOS_ARCHIVO, COLUMNAS_RESULTADO, cierres)
    print(
        f"Nuevas capturas: {len(nuevas)} | Resultados añadidos: {len(cierres)} | "
        f"Series Kalshi fallidas: {len(errores_series)}/{total_series}"
    )
    if errores_series:
        print("Series fallidas:", ", ".join(errores_series))


if __name__ == "__main__":
    main()
