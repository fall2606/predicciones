"""Append immutable Champions League forecast snapshots and a separate settlement ledger."""

import ast
import csv
import os
from datetime import datetime, timedelta, timezone
import requests

from champions_engine import (
    TZ,
    PREDICCIONES_ARCHIVO,
    RESULTADOS_ARCHIVO,
    cargar_eventos_kalshi_champions,
    cargar_partidos_y_estadisticas,
    calibrar_resumen,
    construir_prediccion_partido,
    evaluar_yes_goles,
    preparar_top_predicciones_kalshi,
    resultado_del_periodo,
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

    modelo = cargar_partidos_y_estadisticas(clave, temporada, anterior)
    corners = None
    predicciones, resultados = leer_historial_fijo()
    anexar_csv(PREDICCIONES_ARCHIVO, COLUMNAS_PRED, [])
    anexar_csv(RESULTADOS_ARCHIVO, COLUMNAS_RESULTADO, [])
    try:
        eventos, errores_series, total_series = cargar_eventos_kalshi_champions()
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        eventos, errores_series, total_series = [], [f"No se pudo listar Champions League: {exc}"], 0
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

    partidos_por_id = {str(m["id"]): m for m in modelo[0]}
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
        if serie in {"KXUCLCORNERS", "KXUCLTCORNERS"}:
            if datos["corners_final"] is None:
                continue
            cuenta = (
                sum(datos["corners_final"])
                if serie == "KXUCLCORNERS"
                else datos["corners_final"][0] if market_key[1] == "local"
                else datos["corners_final"][1]
            )
            linea, direccion = market_key[2], market_key[3]
            acierto_yes = cuenta < linea if direccion == "under" else cuenta >= linea
        else:
            marcador = resultado_del_periodo(market_key, datos["resultado"])
            acierto_yes = evaluar_yes_goles(market_key, marcador)
        if acierto_yes is None:
            if serie in {"KXUCLFTTS", "KXUCLFIRSTGOAL"}:
                estado = "UNKNOWN"
            else:
                continue
        else:
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
