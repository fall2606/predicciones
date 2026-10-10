"""Liquida pronósticos archivados de Champions sin recalcular ni sobrescribir capturas."""
import ast
import csv
import os
from datetime import datetime, timezone

import requests

from champions_engine import (
    TZ,
    PREDICCIONES_ARCHIVO,
    RESULTADOS_ARCHIVO,
    cargar_partidos_y_estadisticas,
    construir_prediccion_partido,
    evaluar_yes_goles,
    resultado_del_periodo,
    leer_historial_fijo,
)

COLUMNAS_RESULTADO = ["prediction_id", "estado", "marcador", "resuelto_en"]


def anexar_resultados(filas):
    nuevo = not RESULTADOS_ARCHIVO.exists() or RESULTADOS_ARCHIVO.stat().st_size == 0
    if not filas and not nuevo:
        return
    with RESULTADOS_ARCHIVO.open("a", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=COLUMNAS_RESULTADO, extrasaction="ignore")
        if nuevo:
            writer.writeheader()
        writer.writerows(filas)


def main():
    clave = os.environ["CLAVE"]
    ahora = datetime.now(TZ)
    temporada = ahora.year if ahora.month >= 7 else ahora.year - 1
    modelo = cargar_partidos_y_estadisticas(clave, temporada, temporada - 1)
    predicciones, resultados = leer_historial_fijo()
    if predicciones.empty:
        print("No hay pronósticos de Champions archivados.")
        return

    resueltos = set(resultados["prediction_id"].astype(str)) if not resultados.empty else set()
    pendientes = predicciones[~predicciones["prediction_id"].astype(str).isin(resueltos)]
    partidos = {str(p["id"]): p for p in modelo[0]}
    nuevos = []
    partidos_revisados = set()

    for _, fila in pendientes.iterrows():
        match_id = str(fila.get("match_id", ""))
        partido = partidos.get(match_id)
        if not partido or partido.get("status") != "FINISHED":
            continue
        datos = construir_prediccion_partido(partido, modelo, None)
        resultado = datos.get("resultado")
        if not resultado:
            nuevos.append({
                "prediction_id": str(fila["prediction_id"]),
                "estado": "NO_EVALUABLE",
                "marcador": "Marcador oficial no disponible en la API",
                "resuelto_en": datetime.now(timezone.utc).isoformat(),
            })
            resueltos.add(str(fila["prediction_id"]))
            continue
        try:
            market_key, lado = ast.literal_eval(str(fila.get("clave_interna", "")))
        except (ValueError, SyntaxError, TypeError):
            continue

        serie = market_key[0]
        motivo_no_evaluable = ""
        if serie in {"KXUCLCORNERS", "KXUCLTCORNERS"}:
            if datos.get("corners_final") is None:
                motivo_no_evaluable = "Falta fuente oficial de córners"
                acierto_yes = None
            else:
                if serie == "KXUCLCORNERS":
                    cuenta = sum(datos["corners_final"])
                else:
                    cuenta = datos["corners_final"][0] if market_key[1] == "local" else datos["corners_final"][1]
                linea, direccion = market_key[2], market_key[3]
                acierto_yes = cuenta < linea if direccion == "under" else cuenta >= linea
        else:
            marcador_periodo = resultado_del_periodo(market_key, resultado)
            acierto_yes = evaluar_yes_goles(market_key, marcador_periodo)
            if acierto_yes is None:
                motivo_no_evaluable = (
                    "Falta marcador oficial del descanso"
                    if marcador_periodo is None and serie.startswith("KXUCL1H")
                    else "El marcador final no basta para determinar el primer goleador"
                    if serie in {"KXUCLFTTS", "KXUCLFIRSTGOAL"}
                    else "Faltan datos oficiales para resolver este mercado"
                )
        # Cerrar el partido: WIN/LOSS con evidencia suficiente, o estado explícito
        # cuando la fuente oficial no aporta la estadística concreta. Nunca inventar.
        if acierto_yes is None:
            estado = "NO_EVALUABLE"
        else:
            acierto = acierto_yes if lado == "YES" else not acierto_yes
            estado = "WIN" if acierto else "LOSS"
        marcador = resultado["partido"]
        nuevos.append({
            "prediction_id": str(fila["prediction_id"]),
            "estado": estado,
            "marcador": f"{marcador[0]}-{marcador[1]}" + (f" · {motivo_no_evaluable}" if motivo_no_evaluable else ""),
            "resuelto_en": datetime.now(timezone.utc).isoformat(),
        })
        resueltos.add(str(fila["prediction_id"]))
        partidos_revisados.add(match_id)

    anexar_resultados(nuevos)
    print(
        f"Partidos finalizados revisados: {len(partidos_revisados)} | "
        f"Pronósticos liquidados: {len(nuevos)} | "
        f"Pendientes de resultado/datos oficiales: {len(pendientes) - len(nuevos)}"
    )


if __name__ == "__main__":
    main()
