"""Append immutable Champions League forecast snapshots and a separate settlement ledger."""

import ast
import csv
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import requests
from prediction_core import append_immutable_predictions, rank_market_candidates, extract_line, calculate_edge

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
    "mercado", "line", "seleccion", "bloque", "edge", "clave_interna", "capturado_en",
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
    try:
        eventos, errores_series, total_series = cargar_eventos_kalshi_champions()
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        eventos, errores_series, total_series = [], [f"No se pudo listar Champions League: {exc}"], 0
    market_title_by_ticker = {
        str(mercado.get("ticker")): str(mercado.get("yes_sub_title") or mercado.get("title") or mercado.get("ticker") or "")
        for evento in eventos for mercado in evento.get("markets", [])
        if mercado.get("ticker")
    }
    ids_existentes = set(predicciones["prediction_id"].astype(str)) if not predicciones.empty else set()
    ids_resueltos = set(resultados["prediction_id"].astype(str)) if not resultados.empty else set()

    nuevas = []
    captura = datetime.now(timezone.utc)
    sufijo_captura = captura.strftime("%Y%m%d%H")
    limite = datetime.combine(ahora.date() + timedelta(days=14), datetime.max.time(), tzinfo=TZ)
    for partido in modelo[0]:
        if partido.get("status") not in {"SCHEDULED", "TIMED"}:
            continue
        inicio = datetime.fromisoformat(partido["utcDate"].replace("Z", "+00:00"))
        if not (ahora < inicio.astimezone(TZ) <= limite):
            continue
        datos = construir_prediccion_partido(partido, modelo, corners)
        fecha = datos["fecha"].date()
        salida_kalshi = preparar_top_predicciones_kalshi(
            eventos, datos["local"], datos["visita"], fecha,
            datos["p_goles"], datos["i"], datos["j"],
            datos["p_corners"], datos["ci"], datos["cj"], None, None,
            datos["proporcion_primera"],
        )
        sencillas = calibrar_resumen(salida_kalshi[0], predicciones, resultados)
        alta, valor = rank_market_candidates(
            salida_kalshi[6], min_probability=float(os.getenv("PROBABILIDAD_MINIMA_COMBO", "0.65"))
        )
        resultado_actual = []
        edge_minimo_sencillas = float(os.getenv("SENCILLAS_EDGE_MIN", "0.05"))
        for fila in sencillas:
            fila = dict(fila)
            fila["_bloque"] = "SENCILLAS"
            try:
                _, lado = ast.literal_eval(fila["Clave interna"])
            except (ValueError, SyntaxError, TypeError):
                lado = ""
            fila["_lado"] = lado
            # Recalculate edge from the calibrated probability and actual buy price.
            # Never persist a recommended single when either value is unavailable.
            try:
                p_calibrada = float(str(fila.get("Probabilidad modelo", "")).rstrip("%")) / 100
            except (TypeError, ValueError):
                p_calibrada = None
            fila["edge"] = calculate_edge(p_calibrada, fila.get("Precio de compra"))
            if fila["edge"] is None or fila["edge"] < edge_minimo_sencillas:
                continue
            resultado_actual.append(fila)
        for bloque, candidatas in (("ALTA_PROBABILIDAD", alta), ("A_TENER_EN_CUENTA", valor)):
            for candidato in candidatas:
                try:
                    clave = candidato.get("_clave")
                    lado = candidato.get("_lado") or candidato.get("Lado Kalshi")
                    ptxt = candidato.get("Probabilidad del modelo", "0%")
                    pbase = float(str(ptxt).rstrip("%")) / 100
                except (TypeError, ValueError):
                    continue
                tipo_calibracion = f"{bloque} · {candidato.get('Categoría resumen', 'Kalshi')}"
                calibrado = calibrar_resumen([{
                    "Tipo": tipo_calibracion,
                    "Probabilidad base": pbase,
                    "Probabilidad modelo": f"{pbase:.1%}",
                    "Precio de compra": candidato.get("Precio de compra", "—"),
                }], predicciones, resultados)[0]
                p_calibrada = float(str(calibrado.get("Probabilidad modelo", f"{pbase:.1%}")).rstrip("%")) / 100
                edge_calibrado = calculate_edge(p_calibrada, candidato.get("Precio de compra"))
                # Calibration can erase an apparent advantage; do not keep a pick
                # when the executable price or recalculated edge is missing/negative.
                if edge_calibrado is None or edge_calibrado < 0.02:
                    continue
                resultado_actual.append({
                    "Tipo": candidato.get("Categoría resumen", "Kalshi"),
                    "Jugada sencilla": candidato.get("Apuesta sencilla") or candidato.get("Apuesta", ""),
                    "Probabilidad base": pbase,
                    "Probabilidad modelo": f"{p_calibrada:.1%}",
                    "Calibración": calibrado.get("Calibración", "Sin calibrar"),
                    "Precio ahora": candidato.get("Precio de compra", "—"),
                    "Disponibilidad": candidato.get("Disponibilidad", ""),
                    "Ticker": candidato.get("Ticker", ""),
                    "Clave interna": repr((clave, lado)),
                    "_bloque": bloque, "_lado": lado, "edge": edge_calibrado,
                })
        for fila in resultado_actual:
            fila["Tipo"] = f"{fila['_bloque']} · {fila['Tipo']}"
        match_id = str(partido["id"])
        existentes_partido = predicciones[predicciones["match_id"].astype(str) == match_id].copy() if not predicciones.empty and "match_id" in predicciones.columns else pd.DataFrame()
        categorias_existentes = set(existentes_partido["categoria"].astype(str)) if not existentes_partido.empty and "categoria" in existentes_partido.columns else set()
        cantidad_alta = int(existentes_partido["categoria"].astype(str).str.startswith("ALTA_PROBABILIDAD · ").sum()) if not existentes_partido.empty and "categoria" in existentes_partido.columns else 0
        cantidad_valor = int(existentes_partido["categoria"].astype(str).str.startswith("A_TENER_EN_CUENTA · ").sum()) if not existentes_partido.empty and "categoria" in existentes_partido.columns else 0
        for fila in resultado_actual:
            prediction_id = f"{match_id}|{fila.get('_bloque', 'SENCILLAS')}|{fila.get('Ticker', '')}|{fila.get('_lado', '')}"
            # Legacy captures from earlier code included a timestamp suffix; recognize them
            # as the same market/block so migration never creates duplicate snapshots.
            ids_base_existentes = {"|".join(str(pid).split("|")[:4]) for pid in ids_existentes}
            if prediction_id in ids_base_existentes:
                continue
            bloque = fila.get("_bloque", "SENCILLAS")
            categoria_completa = fila["Tipo"]
            if bloque == "SENCILLAS":
                if categoria_completa in categorias_existentes:
                    continue
            elif bloque == "ALTA_PROBABILIDAD":
                if cantidad_alta >= 5:
                    continue
            elif bloque == "A_TENER_EN_CUENTA":
                if cantidad_valor >= 10:
                    continue
            market_key, lado = ast.literal_eval(fila["Clave interna"])
            nuevas.append({
                "prediction_id": prediction_id,
                "match_id": match_id,
                "fecha": inicio.astimezone(TZ).isoformat(),
                "local": datos["local"],
                "visita": datos["visita"],
                "categoria": categoria_completa,
                "jugada": fila["Jugada sencilla"],
                "probabilidad_base": f"{float(fila['Probabilidad base']):.6f}",
                "probabilidad_calibrada": f"{float(fila['Probabilidad modelo'].rstrip('%')) / 100:.6f}",
                "calibracion": fila["Calibración"],
                "precio": fila["Precio ahora"],
                "disponibilidad": fila["Disponibilidad"],
                "ticker": fila.get("Ticker", "—"),
                "mercado": market_title_by_ticker.get(str(fila.get("Ticker", "")), str(fila.get("Jugada sencilla", ""))),
                "line": extract_line(market_title_by_ticker.get(str(fila.get("Ticker", "")), str(fila.get("Jugada sencilla", "")))) or (str(market_key[2]) if isinstance(market_key, (tuple, list)) and len(market_key) > 2 and market_key[2] is not None else ""),
                "seleccion": lado,
                "bloque": bloque,
                "edge": "" if fila.get("edge") is None else f"{float(fila.get('edge')):.6f}",
                "clave_interna": repr((market_key, lado)),
                "capturado_en": captura.isoformat(),
            })
            ids_existentes.add(prediction_id)
            ids_existentes.add(prediction_id + "|" + sufijo_captura)
            categorias_existentes.add(categoria_completa)
            if bloque == "ALTA_PROBABILIDAD":
                cantidad_alta += 1
            elif bloque == "A_TENER_EN_CUENTA":
                cantidad_valor += 1

    captura_resumen = append_immutable_predictions(PREDICCIONES_ARCHIVO, COLUMNAS_PRED, nuevas, probability_field="probabilidad_calibrada")
    print(f"Capturas nuevas propuestas: {len(nuevas)} | aceptadas: {captura_resumen['accepted']} | descartadas: {captura_resumen['rejected']}. El capturador no escribe resultados.")


if __name__ == "__main__":
    main()
