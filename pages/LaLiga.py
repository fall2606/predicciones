import os
from datetime import datetime, timedelta, timezone
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil, floor
import re
from unicodedata import normalize
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from app_metadata import mostrar_metadata
from scipy.stats import nbinom, poisson


from laliga_engine import *
from prediction_core import rank_market_candidates, rank_single_candidates
from reconciliacion_ui import mostrar_reconciliacion, mostrar_actualizacion_archivos
from pronosticos_ui import mostrar_pronosticos_sencillos, mostrar_tarjetas_mercado
from football_match_utils import apply_reconciliation_statuses, is_upcoming_match, merge_match_sources

 
def _con_resultado_pendiente(filas, estado_partido):
    """Añade Resultado a cualquier tabla de pronósticos sin fingir una liquidación."""
    df = filas.copy() if isinstance(filas, pd.DataFrame) else pd.DataFrame(filas)
    if "Resultado" not in df.columns:
        df["Resultado"] = "ANULADA" if estado_partido in {"CANCELLED", "CANCELED", "POSTPONED"} else ("sin datos para liquidar" if estado_partido in {"FINISHED", "AWARDED"} else "PENDIENTE")
    return df


def _tabla_con_resultado(df, columna="Resultado"):
    """Presenta resultados de forma consistente en el historial y en los tres bloques."""
    if not isinstance(df, pd.DataFrame) or columna not in df.columns:
        return df
    def _color(valor):
        texto = str(valor or "").strip().upper()
        if texto == "LOSS" or "NO SE CUMPLIÓ" in texto:
            return "background-color: #f8d7da; color: #721c24; font-weight: bold"
        if texto == "WIN" or "SE CUMPLIÓ" in texto:
            return "background-color: #d4edda; color: #155724; font-weight: bold"
        if "ANULADA" in texto or "VOID" in texto:
            return "background-color: #dcd9ff; color: #37306b; font-weight: bold"
        if "SIN LIQUIDAR" in texto:
            return "background-color: #f8d7da; color: #721c24; font-weight: bold"
        if "PENDIENTE" in texto or "NO EVALUABLE" in texto:
            return "background-color: #e2e3e5; color: #383d41; font-weight: bold"
        return ""
    return df.style.map(_color, subset=[columna])

st.set_page_config(page_title="LaLiga · predicciones", page_icon="⚽", layout="wide")
st.title("🇪🇸 LaLiga · pronósticos y mercados")
mostrar_metadata()
st.page_link("pages/Contexto_deportivo.py", label="Analizar noticias, lesiones y contexto antes de decidir", icon="🧠")
st.caption(
    "Pronósticos de LaLiga para partidos futuros, con historial inmutable. Las probabilidades se muestran como estimaciones no validadas cuando no hay muestra suficiente para calibrarlas."
)
if st.button("Actualizar datos ahora", key="actualizar_datos_laliga"):
    st.cache_data.clear()
    st.rerun()
st.link_button("Abrir Kalshi", "https://kalshi.com/")


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
predicciones_guardadas, resultados_guardados = leer_historial_fijo()
mostrar_actualizacion_archivos([("Predicciones", PREDICCIONES_ARCHIVO), ("Resultados", RESULTADOS_ARCHIVO), ("Reconciliación", os.path.join(os.path.dirname(os.path.dirname(__file__)), "reconciliacion_pendientes.csv"))])
ids_con_captura = (
    set(predicciones_guardadas["match_id"].astype(str))
    if not predicciones_guardadas.empty else set()
)
if corners is None:
    st.info("Kalshi puede ofrecer mercados de corners; sin estadísticas históricas suficientes, se mostrarán sus contratos pero el modelo no les asignará probabilidad.")

inicio, fin = hoy - timedelta(days=2), hoy + timedelta(days=3)
seleccionables = []
partidos_disponibles = merge_match_sources(modelo[9], partidos)
_ruta_reconciliacion_global = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reconciliacion_pendientes.csv")
partidos_disponibles = apply_reconciliation_statuses(partidos_disponibles, _ruta_reconciliacion_global, "LaLiga")

mostrar_reconciliacion("LaLiga")

for partido in partidos_disponibles:
    fecha_txt = partido.get("utcDate")
    if not fecha_txt:
        continue
    fecha_utc = datetime.fromisoformat(fecha_txt.replace("Z", "+00:00"))
    fecha_local = fecha_utc.astimezone(TZ).date()
    estado = partido.get("status")
    proximo = inicio <= fecha_local <= fin and is_upcoming_match(partido)
    finalizado_revisable = (
        inicio <= fecha_local <= hoy
        and estado in {"FINISHED", "AWARDED", "CANCELLED", "CANCELED", "POSTPONED", "SUSPENDED"}
        and str(partido.get("id")) in ids_con_captura
    )
    if proximo or finalizado_revisable:
        seleccionables.append(partido)

if not seleccionables:
    st.info(f"No hay partidos próximos ni capturas finalizadas para comparar (hoy {inicio:%d/%m} a {fin:%d/%m}).")
    st.stop()

seleccionables.sort(key=lambda m: m.get("utcDate", ""))

# El pronóstico estadístico se muestra para el partido elegido más abajo,
# evitando un segundo selector de partido que pudiera contradecir la vista principal.

vista_partidos = st.radio(
    "Vista",
    ["Próximos", "Auditoría de partidos cerrados/pasados"],
    horizontal=True,
    key="vista_partidos_laliga",
)
if vista_partidos == "Próximos":
    partidos_en_vista = [m for m in seleccionables if is_upcoming_match(m)]
    etiqueta_selector = "Partido próximo"
else:
    partidos_en_vista = [m for m in seleccionables if not is_upcoming_match(m)]
    etiqueta_selector = "Partido cerrado/pasado · auditoría"
if not partidos_en_vista:
    st.info("No hay partidos en esta vista. Los partidos finalizados no se incluyen en Próximos; cambia a Auditoría para revisar capturas anteriores.")
    st.stop()

opciones = {}

for m in partidos_en_vista:
    local = m["homeTeam"].get("shortName") or m["homeTeam"]["name"]
    visita = m["awayTeam"].get("shortName") or m["awayTeam"]["name"]
    fecha = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).astimezone(TZ)
    estado_partido = str(m.get("status", "")).upper()
    estado = (
        "⏳ PRÓXIMO" if is_upcoming_match(m)
        else "✅ FINAL" if estado_partido in {"FINISHED", "AWARDED", "FINAL", "COMPLETED"}
        else "⚠️ " + estado_partido if estado_partido in {"CANCELLED", "CANCELED", "POSTPONED", "SUSPENDED"}
        else "📁 REVISAR"
    )
    marcador = m.get("score", {}).get("fullTime", {})
    marcador_txt = (
        f" · {marcador['home']}-{marcador['away']}"
        if estado == "✅ FINAL" and marcador.get("home") is not None and marcador.get("away") is not None
        else ""
    )
    etiqueta = f"{estado} · {fecha:%a %d/%m %H:%M} · {local} vs {visita}{marcador_txt}"
    opciones[etiqueta] = m

st.caption(
    f"Ventana visible: {inicio:%d/%m}–{fin:%d/%m} (2 días anteriores, hoy y 3 siguientes) · los partidos pasados solo aparecen si tienen predicción guardada · hora de Chicago"
)
pendientes = [m for m in partidos_en_vista if is_upcoming_match(m)]
indice = 0
if pendientes:
    indice = next(i for i, m in enumerate(partidos_en_vista) if m is pendientes[0])
etiqueta = st.selectbox(etiqueta_selector, list(opciones), index=indice)
partido = opciones[etiqueta]
estado_partido = str(partido.get("status", "")).upper()
# El calendario de la app y el informe de liquidación pueden actualizarse en instantes
# distintos. Si la reconciliación oficial ya marcó este partido como final, no mostrarlo
# como próximo ni dejar las tablas en PENDIENTE por un estado antiguo del calendario.
try:
    _ruta_reconciliacion = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reconciliacion_pendientes.csv")
    if os.path.exists(_ruta_reconciliacion):
        _reconciliacion_partidos = pd.read_csv(_ruta_reconciliacion, dtype=str).fillna("")
        _filas_partido = _reconciliacion_partidos[
            _reconciliacion_partidos["deporte"].eq("LaLiga")
            & _reconciliacion_partidos["match_id"].astype(str).eq(str(partido.get("id", "")))
        ]
        _estados_informe = set(_filas_partido["estado_partido"].astype(str).str.upper())
        if _estados_informe & {"FINISHED", "AWARDED", "FINAL", "COMPLETED"}:
            estado_partido = "FINISHED"
            partido["status"] = "FINISHED"
        elif _estados_informe & {"CANCELLED", "CANCELED"}:
            estado_partido = "CANCELLED"
            partido["status"] = "CANCELLED"
except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError):
    pass
partido_final = estado_partido in {"FINISHED", "AWARDED"}
partido_cerrado = partido_final or estado_partido in {"CANCELLED", "CANCELED", "POSTPONED", "SUSPENDED"}
local = partido["homeTeam"].get("shortName") or partido["homeTeam"]["name"]
visita = partido["awayTeam"].get("shortName") or partido["awayTeam"]["name"]
fecha_objetivo = datetime.fromisoformat(partido["utcDate"].replace("Z", "+00:00")).astimezone(TZ)
datos_partido = construir_prediccion_partido(partido, modelo, corners)
local, visita = datos_partido["local"], datos_partido["visita"]
fecha_objetivo, modelo_objetivo = datos_partido["fecha"], datos_partido["modelo"]
gl, gv = datos_partido["gl"], datos_partido["gv"]
ajuste_forma_local, ajuste_forma_visita = datos_partido["ajuste_local"], datos_partido["ajuste_visita"]
p_goles, i, j = datos_partido["p_goles"], datos_partido["i"], datos_partido["j"]

if not partido_cerrado:
    p_local_modelo = float(p_goles[i > j].sum())
    p_empate_modelo = float(p_goles[i == j].sum())
    p_visita_modelo = float(p_goles[i < j].sum())
    st.subheader("Pronóstico del partido seleccionado")
    with st.container(border=True):
        st.markdown(f"**{local} vs {visita}**")
        st.caption(f"Marcador medio estimado: {gl:.2f}–{gv:.2f} goles · cálculo basado en las estadísticas disponibles de la competición")
        c_local, c_empate, c_visita = st.columns(3)
        c_local.metric("Gana local · modelo", f"{p_local_modelo:.1%}")
        c_empate.metric("Empate · modelo", f"{p_empate_modelo:.1%}")
        c_visita.metric("Gana visitante · modelo", f"{p_visita_modelo:.1%}")
        st.caption("Estas son probabilidades del modelo; la probabilidad final para apostar solo se calcula con un precio de compra válido de Kalshi.")
p_corners, ci, cj = datos_partido["p_corners"], datos_partido["ci"], datos_partido["cj"]
media_corners_local = datos_partido["media_corners_local"]
media_corners_visita = datos_partido["media_corners_visita"]
corners_final, resultado = datos_partido["corners_final"], datos_partido["resultado"]
try:
    eventos_kalshi, series_kalshi_con_error, total_series_kalshi = cargar_eventos_kalshi_laliga()
    fecha_partido = datetime.fromisoformat(partido["utcDate"].replace("Z", "+00:00")).astimezone(TZ).date()
    resultado_en_vivo = resultado if not partido_cerrado else None
    corners_en_vivo = corners_final if not partido_cerrado else None
    resumen_sencillo_kalshi, margen_yes_kalshi, margen_no_kalshi, candidatos_yes_kalshi, candidatos_no_kalshi, top_kalshi, todas_kalshi, mercados_kalshi, num_plantillas_kalshi = preparar_top_predicciones_kalshi(
        eventos_kalshi, local, visita, fecha_partido, p_goles, i, j, p_corners, ci, cj,
        resultado_en_vivo, corners_en_vivo, datos_partido["proporcion_primera"],
    )
    resumen_sencillo_kalshi = calibrar_resumen(resumen_sencillo_kalshi, predicciones_guardadas, resultados_guardados)
    # Aplica la calibración de categoría antes de anclar también a las listas diagnósticas.
    todas_kalshi = calibrar_resumen(todas_kalshi, predicciones_guardadas, resultados_guardados)
    margen_yes_kalshi = calibrar_resumen(margen_yes_kalshi, predicciones_guardadas, resultados_guardados)
    margen_no_kalshi = calibrar_resumen(margen_no_kalshi, predicciones_guardadas, resultados_guardados)
    candidatos_yes_kalshi = calibrar_resumen(candidatos_yes_kalshi, predicciones_guardadas, resultados_guardados)
    candidatos_no_kalshi = calibrar_resumen(candidatos_no_kalshi, predicciones_guardadas, resultados_guardados)
    top_kalshi = calibrar_resumen(top_kalshi, predicciones_guardadas, resultados_guardados)
except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
    eventos_kalshi, series_kalshi_con_error, total_series_kalshi = [], [], 0
    resumen_sencillo_kalshi, margen_yes_kalshi, margen_no_kalshi, candidatos_yes_kalshi, candidatos_no_kalshi, top_kalshi, todas_kalshi, mercados_kalshi, num_plantillas_kalshi = [], [], [], [], [], [], [], [], 0
    st.error(f"No pude consultar los mercados abiertos de Kalshi: {exc}")

resumen_archivado = resumen_guardado(partido["id"], predicciones_guardadas, resultados_guardados)
if resumen_archivado and partido_cerrado:
    for _jugada_archivada in resumen_archivado:
        _resultado_actual = str(_jugada_archivada.get("Resultado", "")).upper()
        if estado_partido in {"CANCELLED", "CANCELED", "POSTPONED"}:
            _jugada_archivada["Resultado"] = "ANULADA"
        elif not any(token in _resultado_actual for token in ("WIN", "LOSS", "ANULADA")):
            _jugada_archivada["Resultado"] = "sin datos para liquidar · mercado sin fuente compatible"
if resumen_archivado:
    resumen_sencillo_kalshi = resumen_archivado
elif partido_cerrado:
    resumen_sencillo_kalshi = []

st.subheader(f"{local} vs {visita}")
if partido_final and resultado is not None:
    st.success(
        f"Partido finalizado: {local} {resultado['partido'][0]}–{resultado['partido'][1]} {visita}. "
        "El resultado de cada pronóstico archivado aparece en su tarjeta."
    )
    st.caption("Las capturas y los resultados se guardan por separado; se muestra el último pronóstico archivado antes del inicio, sin reescribir los anteriores.")
st.markdown(
    f"**Forma reciente · últimos 8 partidos de liga antes del encuentro**  \n"
    f"{local}: {resumen_forma(local, modelo_objetivo, fecha_objetivo)}  ·  "
    f"{visita}: {resumen_forma(visita, modelo_objetivo, fecha_objetivo)}"
)
st.markdown("### Contratos reales de Kalshi · posibles apuestas")
st.caption("Esta sección sí contiene contratos negociables de Kalshi. Si no hay contratos disponibles, usa el pronóstico del modelo de arriba; no son lo mismo.")
if resumen_archivado:
    st.caption("Se muestra la última captura guardada antes del inicio; las anteriores permanecen inmutables y se pueden comparar abajo.")
elif partido_cerrado:
    st.caption("No hay una captura previa guardada para este encuentro cerrado; no se inventa una predicción retrospectiva.")
else:
    st.caption("Estimación en vivo. La captura automática se guarda cada dos horas; la hora de la última captura se muestra cuando esté disponible.")
# Salvaguarda también las capturas históricas: las sencillas siempre respetan el filtro de valor.
resumen_sencillo_kalshi = rank_single_candidates(resumen_sencillo_kalshi, limit=5)
if resumen_sencillo_kalshi:
    for jugada in resumen_sencillo_kalshi:
        with st.container(border=True):
            st.markdown(f"**{jugada.get('Tipo', 'Sencilla')}:** {jugada.get('Jugada sencilla', '—')}")
            st.write(f"Probabilidad del modelo calibrada: **{jugada.get('Probabilidad del modelo', jugada.get('Probabilidad modelo', '—'))}**")
            st.write(f"Probabilidad final anclada a Kalshi: **{jugada.get('Probabilidad final', 'Sin precio válido')}**")
            precio_txt = jugada.get("Precio de compra", jugada.get("Precio ahora", "—"))
            st.write(f"Precio de compra Kalshi: **{precio_txt}**")
            try:
                edge_num = float(jugada.get("Ventaja (edge)", 0))
                if abs(edge_num) > 1:
                    edge_num /= 100
                edge_txt = f"{edge_num * 100:+.1f} puntos"
            except (TypeError, ValueError):
                edge_txt = "No disponible"
            st.write(f"Ventaja: **{edge_txt}**")
            mercado_texto = str(jugada.get("Jugada sencilla") or jugada.get("Apuesta") or jugada.get("Tipo") or "")
            fundamento = (f"El modelo proyecta {local} {gl:.2f} goles y {visita} {gv:.2f}. "
                          "Esta proyección es el punto de partida; por sí sola no justifica una apuesta si el mercado y la línea no encajan.")
            try:
                linea_contexto = float(jugada.get("line"))
                if any(token in mercado_texto.casefold() for token in ("más de", "over", "total goles")):
                    total_esperado = gl + gv
                    fundamento += f" Total esperado: {total_esperado:.2f} frente a la línea {linea_contexto:g}."
            except (TypeError, ValueError):
                fundamento += " No hay una línea numérica legible en la tarjeta para contrastar el umbral."
            st.caption("Fundamento del modelo: " + fundamento)
            if jugada.get("Revisión") == "REVISAR":
                st.warning("REVISAR · ventaja superior a 20 puntos; comprobar precio y modelo.")
            resultado_tarjeta = str(jugada.get("Resultado", "PENDIENTE")).upper()
            if "ANULADA" in resultado_tarjeta or "VOID" in resultado_tarjeta:
                st.info("ANULADA")
            elif resultado_tarjeta == "WIN" or "SE CUMPLIÓ" in resultado_tarjeta:
                st.success("WIN")
            elif resultado_tarjeta == "LOSS" or "NO SE CUMPLIÓ" in resultado_tarjeta:
                st.error("LOSS")
            elif "SIN DATOS PARA LIQUIDAR" in resultado_tarjeta:
                st.caption(resultado_tarjeta)
            elif "SIN LIQUIDAR" in resultado_tarjeta:
                st.error("SIN LIQUIDAR · revisar liquidación del contrato")
            else:
                st.caption("PENDIENTE" if not partido_cerrado else "sin datos para liquidar · mercado sin fuente compatible")
            st.caption(f"Ticker: {jugada.get('Ticker', '—')}")
    if resumen_archivado:
        historial_partido = historial_guardado(partido["id"], predicciones_guardadas, resultados_guardados)
        if not historial_partido.empty and partido_cerrado:
            if estado_partido in {"CANCELLED", "CANCELED", "POSTPONED"}:
                historial_partido["Resultado"] = "ANULADA"
            else:
                _estado_historial = historial_partido["Resultado"].astype(str).str.upper()
                _terminal_historial = _estado_historial.str.contains("WIN|LOSS|ANULADA", regex=True)
                historial_partido.loc[~_terminal_historial, "Resultado"] = "sin datos para liquidar · mercado sin fuente compatible"
        with st.expander(f"Auditoría · capturas guardadas ({len(historial_partido)})"):
            st.dataframe(_tabla_con_resultado(historial_partido), use_container_width=True, hide_index=True)
else:
    if partido_cerrado:
        st.info("No hay una captura previa de este encuentro cerrado; no se inventa una comparación retrospectiva.")
    else:
        st.info("No encontré mercados modelables de Kalshi para resumir este partido.")


st.markdown("### 5 de alta probabilidad para posibles combos")
st.caption("Ranking del catálogo real de Kalshi. Se omiten contratos sin ticker o sin probabilidad válida y se evita combinar familias de mercados correlacionadas; si no hay cinco independientes, no se rellena artificialmente.")
alta_vista, valor_vista = rank_market_candidates(
    todas_kalshi, min_probability=float(os.getenv("PROBABILIDAD_MINIMA_COMBO", "0.65"))
)
if partido_cerrado:
    _archivo_partido = predicciones_guardadas[predicciones_guardadas["match_id"].astype(str) == str(partido["id"])].copy() if not predicciones_guardadas.empty else pd.DataFrame()
    _mask_alta = _archivo_partido["bloque"].astype(str).eq("ALTA_PROBABILIDAD") if not _archivo_partido.empty and "bloque" in _archivo_partido.columns else pd.Series(False, index=_archivo_partido.index)
    if not _archivo_partido.empty and "categoria" in _archivo_partido.columns:
        _mask_alta = _mask_alta | _archivo_partido["categoria"].astype(str).str.startswith("ALTA_PROBABILIDAD · ")
    _archivo_partido = _archivo_partido[_mask_alta].copy() if not _archivo_partido.empty else pd.DataFrame()
    if not _archivo_partido.empty:
        _res = resultados_guardados.sort_values("resuelto_en").drop_duplicates("prediction_id", keep="last") if not resultados_guardados.empty and "resuelto_en" in resultados_guardados.columns else resultados_guardados
        _archivo_partido = _archivo_partido.merge(_res[["prediction_id", "estado"]], on="prediction_id", how="left")
        _archivo_partido["Resultado"] = _archivo_partido["estado"].fillna("ANULADA" if estado_partido in {"CANCELLED", "CANCELED", "POSTPONED"} else ("sin datos para liquidar" if partido_cerrado else "PENDIENTE"))
        _sin_ticker_alta = _archivo_partido["ticker"].astype(str).str.strip().str.upper().isin(["", "—", "-", "NONE", "NAN"])
        _archivo_partido.loc[_archivo_partido["estado"].isna() & _sin_ticker_alta, "Resultado"] = "sin datos para liquidar · histórico sin contrato"
        _archivo_partido.loc[_archivo_partido["estado"].astype(str).str.upper().isin(["POSTPONED", "CANCELLED", "CANCELED"]), "Resultado"] = "ANULADA"
        for _col in ["bloque", "mercado", "line", "seleccion", "edge", "precio", "calibracion"]:
            if _col not in _archivo_partido.columns:
                _archivo_partido[_col] = ""
        mostrar_tarjetas_mercado(_archivo_partido[["bloque", "mercado", "line", "seleccion", "jugada", "probabilidad_calibrada", "precio", "edge", "calibracion", "ticker", "Resultado"]], titulo="Pronósticos archivados · no implica que el contrato siga disponible", limite=5)
    else:
        st.info("sin datos para liquidar · no existe captura prepartido guardada en el bloque de alta probabilidad.")
elif alta_vista:
    _alta_df = pd.DataFrame(alta_vista).copy()
    _alta_df["Resultado"] = "ANULADA" if estado_partido in {"CANCELLED", "CANCELED", "POSTPONED"} else ("sin datos para liquidar · mercado sin fuente compatible" if partido_cerrado else "PENDIENTE")
    _cols = [c for c in ["Apuesta", "line", "Lado Kalshi", "Probabilidad del modelo", "Precio de compra", "Ventaja (edge)", "Calibración", "Ticker", "Resultado"] if c in _alta_df.columns]
    mostrar_tarjetas_mercado(_alta_df[_cols], titulo="Contratos de Kalshi que superan los filtros", limite=5)
else:
    st.info("No hay cinco selecciones independientes que superen el umbral configurable.")

st.markdown("### 10 a tener en cuenta · ordenadas por edge")
st.caption("Sólo se incluyen contratos con precio y ventaja calculable. La ventaja es probabilidad del modelo menos precio implícito; no se inventan precios cuando Kalshi no cotiza.")
if partido_cerrado:
    _archivo_valor = predicciones_guardadas[predicciones_guardadas["match_id"].astype(str) == str(partido["id"])].copy() if not predicciones_guardadas.empty else pd.DataFrame()
    _mask_valor = _archivo_valor["bloque"].astype(str).eq("A_TENER_EN_CUENTA") if not _archivo_valor.empty and "bloque" in _archivo_valor.columns else pd.Series(False, index=_archivo_valor.index)
    if not _archivo_valor.empty and "categoria" in _archivo_valor.columns:
        _mask_valor = _mask_valor | _archivo_valor["categoria"].astype(str).str.startswith("A_TENER_EN_CUENTA · ")
    _archivo_valor = _archivo_valor[_mask_valor].copy() if not _archivo_valor.empty else pd.DataFrame()
    if not _archivo_valor.empty:
        _res = resultados_guardados.sort_values("resuelto_en").drop_duplicates("prediction_id", keep="last") if not resultados_guardados.empty and "resuelto_en" in resultados_guardados.columns else resultados_guardados
        _archivo_valor = _archivo_valor.merge(_res[["prediction_id", "estado"]], on="prediction_id", how="left")
        _archivo_valor["Resultado"] = _archivo_valor["estado"].fillna("ANULADA" if estado_partido in {"CANCELLED", "CANCELED", "POSTPONED"} else ("sin datos para liquidar · mercado sin fuente compatible" if partido_cerrado else "PENDIENTE"))
        _sin_ticker = _archivo_valor["ticker"].astype(str).str.strip().str.upper().isin(["", "—", "-", "NONE", "NAN"])
        _archivo_valor.loc[_archivo_valor["estado"].isna() & _sin_ticker, "Resultado"] = "sin datos para liquidar · histórico sin contrato"
        _archivo_valor.loc[_archivo_valor["estado"].astype(str).str.upper().isin(["POSTPONED", "CANCELLED", "CANCELED"]), "Resultado"] = "ANULADA"
        for _col in ["bloque", "mercado", "line", "seleccion", "edge", "calibracion"]:
            if _col not in _archivo_valor.columns:
                _archivo_valor[_col] = ""
        mostrar_tarjetas_mercado(_archivo_valor[["bloque", "mercado", "line", "seleccion", "jugada", "probabilidad_calibrada", "precio", "edge", "calibracion", "ticker", "Resultado"]], titulo="Pronósticos archivados · revisa el resultado antes de comparar", limite=10)
    else:
        st.info("sin datos para liquidar · no existe captura prepartido guardada en el bloque de valor.")
elif valor_vista:
    _valor_df = pd.DataFrame(valor_vista).copy()
    _valor_df["Resultado"] = "ANULADA" if estado_partido in {"CANCELLED", "CANCELED", "POSTPONED"} else ("sin datos para liquidar · mercado sin fuente compatible" if partido_cerrado else "PENDIENTE")
    _ticker_col = "Ticker" if "Ticker" in _valor_df.columns else ("ticker" if "ticker" in _valor_df.columns else None)
    if _ticker_col and estado_partido not in {"CANCELLED", "CANCELED", "POSTPONED"}:
        _sin_ticker = _valor_df[_ticker_col].astype(str).str.strip().str.upper().isin(["", "—", "-", "NONE", "NAN"])
        _valor_df.loc[_sin_ticker, "Resultado"] = "sin datos para liquidar · histórico sin contrato"
    _cols = [c for c in ["Apuesta", "Lado Kalshi", "Probabilidad del modelo", "Precio de compra", "Ventaja (edge)", "Calibración", "Ticker", "Resultado"] if c in _valor_df.columns]
    mostrar_tarjetas_mercado(_valor_df[_cols], titulo="Contratos ordenados por ventaja estimada", limite=10)
else:
    st.info("No hay contratos con precio y edge calculable que entren en este bloque.")

tabla_rendimiento, cantidad_rendimiento = metricas_historial(predicciones_guardadas, resultados_guardados)
with st.expander(f"Auditoría · rendimiento histórico ({cantidad_rendimiento} resultados evaluables)"):
    if cantidad_rendimiento:
        st.dataframe(tabla_rendimiento, use_container_width=True, hide_index=True)
        st.caption("El Brier medio mide el error probabilístico; más bajo es mejor. Las filas originales nunca se reescriben.")
    else:
        st.info("El historial empezará a mostrarse después de guardar pronósticos previos y recibir sus resultados.")

if partido_cerrado:
    st.info("Las cotizaciones y oportunidades actuales se muestran sólo para partidos próximos; arriba queda la captura histórica inmutable.")
else:
    st.markdown("### Apuestas con margen neto estimado")
    st.caption(
        "Solo contratos con precio disponible para este partido, probabilidad estimada ≥50%, ganancia esperada neta ≥$0.05 por contrato "
        "y ROI esperado ≥10%. Separamos YES y NO; las ordenamos por retorno estimado. Si una pestaña queda vacía, el modelo no detectó una opción que cumpla los filtros."
    )
    tab_yes_kalshi, tab_no_kalshi = st.tabs(["YES con margen", "NO con margen"])
    with tab_yes_kalshi:
        if margen_yes_kalshi:
            mostrar_tarjetas_mercado(_con_resultado_pendiente(margen_yes_kalshi, estado_partido), titulo="Contratos YES con margen neto estimado", limite=5)
        else:
            st.info("No hay una apuesta YES disponible ahora que supere el margen mínimo del modelo.")
    with tab_no_kalshi:
        if margen_no_kalshi:
            mostrar_tarjetas_mercado(_con_resultado_pendiente(margen_no_kalshi, estado_partido), titulo="Contratos NO con margen neto estimado", limite=5)
        else:
            st.info("No hay una apuesta NO disponible ahora que supere el margen mínimo del modelo.")
    if not margen_yes_kalshi and not margen_no_kalshi:
        st.info("Ninguna cotización actual pasa el margen mínimo. Abajo puedes ver los mejores precios actuales y el máximo al que cada selección sí cumpliría el objetivo.")

    with st.expander("Ver los mejores candidatos al precio actual y el precio máximo recomendado"):
        st.caption("Estas filas son diagnósticas: si dicen ‘No tiene margen suficiente’, no cumplen el objetivo al precio mostrado. El precio máximo indica dónde podrían alcanzar la ganancia y el ROI mínimos.")
        yes_cand_tab, no_cand_tab = st.tabs(["Candidatos YES", "Candidatos NO"])
        with yes_cand_tab:
            if candidatos_yes_kalshi:
                mostrar_tarjetas_mercado(_con_resultado_pendiente(candidatos_yes_kalshi, estado_partido), titulo="Candidatos YES · no todos cumplen el margen", limite=5)
            else:
                st.info("No hay cotizaciones YES que pueda comparar para este partido.")
        with no_cand_tab:
            if candidatos_no_kalshi:
                mostrar_tarjetas_mercado(_con_resultado_pendiente(candidatos_no_kalshi, estado_partido), titulo="Candidatos NO · no todos cumplen el margen", limite=5)
            else:
                st.info("No hay cotizaciones NO que pueda comparar para este partido.")

    with st.expander("Ver las 10 opciones más probables aunque no tengan margen"):
        if top_kalshi:
            mostrar_tarjetas_mercado(_con_resultado_pendiente(top_kalshi, estado_partido), titulo="Opciones más probables, aunque pueden no tener margen", limite=10)
        else:
            st.info("No encontré tipos y líneas activos de Kalshi en LaLiga con estimación suficiente para este partido.")
st.write(f"**Marcador esperado:** {gl:.1f}–{gv:.1f} goles")
st.caption(
    f"Forma reciente (últimos 8 partidos, ajuste limitado): {local} ×{ajuste_forma_local:.2f} · "
    f"{visita} ×{ajuste_forma_visita:.2f}. El histórico completo también pondera más los resultados recientes."
)

with st.expander("Cómo calcula el algoritmo estas probabilidades"):
    st.markdown(
        "**Goles y actualidad:** estima los goles esperados con fuerzas de ataque/defensa de LaLiga, ponderadas por antigüedad "
        f"(la mitad del peso cada {HALF_LIFE_DAYS} días) y suavizadas con el promedio de liga (K=6). Además aplica un ajuste limitado "
        "por los últimos ocho partidos de cada equipo, usando solo encuentros anteriores a la fecha seleccionada. Luego calcula una distribución Poisson para cada equipo "
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
        "**Primer y segundo tiempo:** usa la proporción de goles antes del descanso observada en partidos previos ponderados por antigüedad; "
        "si faltan datos, usa 45%/55%. Sigue siendo una estimación aproximada porque no modela jugadas ni alineaciones por periodo."
    )
    st.markdown(
        "**Margen:** calcula la probabilidad final anclada al mercado (75% del precio Kalshi + 25% del modelo calibrado por defecto) menos el precio de compra y una tarifa estándar estimada. "
        "Solo recomienda oportunidades con probabilidad ≥50%, ganancia esperada ≥$0.05 y retorno neto esperado ≥10%. "
        "También muestra el precio máximo al que cada selección alcanzaría esos objetivos; si el precio actual es mayor, "
        "la marca como sin margen suficiente. Las tarifas reales pueden variar por mercado y tipo de orden."
    )
    st.caption(
        "Las capturas originales y sus resultados se guardan por separado. Al acumular 30 resultados comparables de una categoría, "
        "la probabilidad futura se calibra con una corrección suavizada; la probabilidad base de cada captura queda intacta."
    )

if not partido_final:
    if todas_kalshi:
        with st.expander(f"Ver todas las apuestas y líneas detectadas ({len(todas_kalshi)})"):
            st.dataframe(pd.DataFrame(todas_kalshi), use_container_width=True, hide_index=True)

    with st.expander("Auditoría · contratos abiertos ahora para este partido"):
        if mercados_kalshi:
            st.dataframe(pd.DataFrame(mercados_kalshi), use_container_width=True, hide_index=True)
        elif total_series_kalshi and not series_kalshi_con_error:
            st.info("Kalshi no tiene contratos abiertos para este encuentro y fecha.")
        else:
            st.info("No pude confirmar los contratos abiertos para este encuentro.")

    if series_kalshi_con_error:
        st.warning(f"Kalshi no respondió para {len(series_kalshi_con_error)} de {total_series_kalshi} series de LaLiga; el listado puede estar incompleto.")
        with st.expander("Auditoría · series de Kalshi que fallaron"):
            st.write(", ".join(series_kalshi_con_error))

    st.caption(
        "Las filas ‘Tipo/línea ofrecido en LaLiga; falta abrirlo para este partido’ usan una línea real detectada en otros partidos de la liga. "
        "Sirven como posibilidades del modelo; Kalshi podría no abrir ese contrato para este encuentro."
    )
st.warning(
    "Las probabilidades son estimaciones y no garantías. Para apostar en Kalshi, verifica que el contrato exacto esté disponible "
    "para este partido y revisa las reglas del mercado."
)
st.info("Los hándicaps del resumen describen el resultado binario YES/NO de Kalshi. No significan devolución por empate exacto con la línea; consulta las reglas del contrato y su ticker antes de operar.")
