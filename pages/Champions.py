from datetime import datetime, timedelta
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
from scipy.stats import nbinom, poisson


from champions_engine import *

st.set_page_config(page_title="Champions League · predicciones", page_icon="🏆", layout="wide")
st.title("🏆 Champions League · mercados y probabilidades de Kalshi")
st.caption(
    "Pronósticos de Champions League para partidos próximos, con comparación inmutable de capturas anteriores y calibración basada en resultados."
)
st.link_button("Abrir Kalshi", "https://kalshi.com/")


try:
    clave_api = st.secrets["CLAVE"]
except Exception:
    st.error("Falta configurar el secreto CLAVE de football-data.org en la app.")
    st.stop()

hoy = datetime.now(TZ).date()
temporada = hoy.year if hoy.month >= 7 else hoy.year - 1
temporada_anterior = temporada - 1

try:
    modelo = cargar_partidos_y_estadisticas(clave_api, temporada, temporada_anterior)
    partidos = modelo[0]
except requests.RequestException as exc:
    st.error(f"No pude cargar Champions League desde football-data.org: {exc}")
    st.stop()
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"No pude preparar el modelo de Champions League: {exc}")
    st.stop()

corners = None
predicciones_guardadas, resultados_guardados = leer_historial_fijo()
ids_con_captura = (
    set(predicciones_guardadas["match_id"].astype(str))
    if not predicciones_guardadas.empty else set()
)
if corners is None:
    st.info("Los mercados de corners que ofrezca Kalshi aparecerán en el detalle, pero esta fuente no tiene estadísticas históricas de corners de Champions para calcular probabilidades.")

inicio, fin = hoy, hoy + timedelta(days=2)
fin_ventana_inicial = fin
inicio_resultados = hoy - timedelta(days=7)
seleccionables = []
partidos_disponibles = {str(m["id"]): m for m in modelo[9]}
partidos_disponibles.update({str(m["id"]): m for m in partidos})
proximos_disponibles = []
for partido in partidos_disponibles.values():
    fecha_txt = partido.get("utcDate")
    if not fecha_txt or partido.get("status") not in {"SCHEDULED", "TIMED"}:
        continue
    fecha_local = datetime.fromisoformat(fecha_txt.replace("Z", "+00:00")).astimezone(TZ).date()
    if fecha_local > fin_ventana_inicial:
        proximos_disponibles.append((fecha_local, partido))

proximos_en_ventana = [
    (fecha, partido) for fecha, partido in proximos_disponibles
    if inicio <= fecha <= fin_ventana_inicial
]
proxima_jornada_automatica = False
if not proximos_en_ventana and proximos_disponibles:
    siguiente_fecha = min(fecha for fecha, _ in proximos_disponibles if fecha > fin_ventana_inicial)
    # Si no hay partido en las próximas 72 horas, enseña los encuentros del siguiente bloque de dos días.
    fin = siguiente_fecha + timedelta(days=1)
    proxima_jornada_automatica = True

for partido in partidos_disponibles.values():
    fecha_txt = partido.get("utcDate")
    if not fecha_txt:
        continue
    fecha_utc = datetime.fromisoformat(fecha_txt.replace("Z", "+00:00"))
    fecha_local = fecha_utc.astimezone(TZ).date()
    estado = partido.get("status")
    proximo = inicio <= fecha_local <= fin and estado in {"SCHEDULED", "TIMED"}
    finalizado_revisable = (
        estado == "FINISHED"
        and (inicio_resultados <= fecha_local <= hoy or str(partido.get("id")) in ids_con_captura)
    )
    if proximo or finalizado_revisable:
        seleccionables.append(partido)

if not seleccionables:
    st.info(f"No hay partidos en el calendario recibido ni capturas finalizadas para comparar. Ventana consultada: {inicio:%d/%m}–{fin:%d/%m} (Chicago).")
    st.stop()
seleccionables.sort(key=lambda m: m.get("utcDate", ""))
opciones = {}
for m in seleccionables:
    local = m["homeTeam"].get("shortName") or m["homeTeam"]["name"]
    visita = m["awayTeam"].get("shortName") or m["awayTeam"]["name"]
    fecha = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).astimezone(TZ)
    estado = "✅ FINAL" if m["status"] == "FINISHED" else "⏳ PRÓXIMO"
    marcador = marcador_reglamentario(m)
    marcador_txt = (
        f" · {marcador['home']}-{marcador['away']}"
        if estado == "✅ FINAL" and marcador.get("home") is not None and marcador.get("away") is not None
        else ""
    )
    etiqueta = f"{estado} · {fecha:%a %d/%m %H:%M} · {local} vs {visita}{marcador_txt}"
    opciones[etiqueta] = m

if proxima_jornada_automatica:
    st.info(
        f"No había partidos hasta el {fin_ventana_inicial:%d/%m}; muestro la siguiente jornada disponible: "
        f"{siguiente_fecha:%d/%m}–{fin:%d/%m}."
    )
st.caption(
    f"Próximos: {inicio:%d/%m}–{fin:%d/%m} · finalizados: últimos 7 días y todos los que tengan captura guardada · hora de Chicago"
)
pendientes = [m for m in seleccionables if m["status"] in {"SCHEDULED", "TIMED"}]
indice = 0
if pendientes:
    indice = next(i for i, m in enumerate(seleccionables) if m is pendientes[0])
etiqueta = st.selectbox("Partido", list(opciones), index=indice)
partido = opciones[etiqueta]
local = partido["homeTeam"].get("shortName") or partido["homeTeam"]["name"]
visita = partido["awayTeam"].get("shortName") or partido["awayTeam"]["name"]
fecha_objetivo = datetime.fromisoformat(partido["utcDate"].replace("Z", "+00:00")).astimezone(TZ)
datos_partido = construir_prediccion_partido(partido, modelo, corners)
local, visita = datos_partido["local"], datos_partido["visita"]
fecha_objetivo, modelo_objetivo = datos_partido["fecha"], datos_partido["modelo"]
gl, gv = datos_partido["gl"], datos_partido["gv"]
ajuste_forma_local, ajuste_forma_visita = datos_partido["ajuste_local"], datos_partido["ajuste_visita"]
p_goles, i, j = datos_partido["p_goles"], datos_partido["i"], datos_partido["j"]
p_corners, ci, cj = datos_partido["p_corners"], datos_partido["ci"], datos_partido["cj"]
media_corners_local = datos_partido["media_corners_local"]
media_corners_visita = datos_partido["media_corners_visita"]
corners_final, resultado = datos_partido["corners_final"], datos_partido["resultado"]
try:
    eventos_kalshi, series_kalshi_con_error, total_series_kalshi = cargar_eventos_kalshi_champions()
    fecha_partido = datetime.fromisoformat(partido["utcDate"].replace("Z", "+00:00")).astimezone(TZ).date()
    resultado_en_vivo = resultado if partido["status"] != "FINISHED" else None
    corners_en_vivo = corners_final if partido["status"] != "FINISHED" else None
    resumen_sencillo_kalshi, margen_yes_kalshi, margen_no_kalshi, candidatos_yes_kalshi, candidatos_no_kalshi, top_kalshi, todas_kalshi, mercados_kalshi, num_plantillas_kalshi = preparar_top_predicciones_kalshi(
        eventos_kalshi, local, visita, fecha_partido, p_goles, i, j, p_corners, ci, cj,
        resultado_en_vivo, corners_en_vivo, datos_partido["proporcion_primera"],
    )
    resumen_sencillo_kalshi = calibrar_resumen(resumen_sencillo_kalshi, predicciones_guardadas, resultados_guardados)
except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
    eventos_kalshi, series_kalshi_con_error, total_series_kalshi = [], [], 0
    resumen_sencillo_kalshi, margen_yes_kalshi, margen_no_kalshi, candidatos_yes_kalshi, candidatos_no_kalshi, top_kalshi, todas_kalshi, mercados_kalshi, num_plantillas_kalshi = [], [], [], [], [], [], [], [], 0
    st.error(f"No pude consultar los mercados abiertos de Kalshi: {exc}")

resumen_archivado = resumen_guardado(partido["id"], predicciones_guardadas, resultados_guardados)
if resumen_archivado:
    resumen_sencillo_kalshi = resumen_archivado
elif partido["status"] == "FINISHED":
    resumen_sencillo_kalshi = []

st.subheader(f"{local} vs {visita}")
if partido["status"] == "FINISHED" and resultado is not None:
    st.success(
        f"Partido finalizado: {local} {resultado['partido'][0]}–{resultado['partido'][1]} {visita}. "
        "El resultado de cada pronóstico archivado aparece en su tarjeta."
    )
    st.caption("Las capturas y los resultados se guardan por separado; se muestra el último pronóstico archivado antes del inicio, sin reescribir los anteriores.")
st.markdown(
    f"**Forma reciente · últimos 8 partidos de Champions antes del encuentro**  \n"
    f"{local}: {resumen_forma(local, modelo_objetivo, fecha_objetivo)}  ·  "
    f"{visita}: {resumen_forma(visita, modelo_objetivo, fecha_objetivo)}"
)
st.markdown("### Resumen sencillo del partido")
if resumen_archivado:
    st.caption("Se muestra la última captura guardada antes del inicio; las anteriores permanecen inmutables y se pueden comparar abajo.")
elif partido["status"] == "FINISHED":
    st.caption("Este partido terminó antes de que se activara el archivo; no hay una predicción previa guardada.")
else:
    st.caption("Estimación en vivo. La captura automática se guarda cada dos horas; la hora de la última captura se muestra cuando esté disponible.")
if resumen_sencillo_kalshi:
    for jugada in resumen_sencillo_kalshi:
        with st.container(border=True):
            st.markdown(f"**{jugada['Tipo']}:** {jugada['Jugada sencilla']}")
            st.write(f"Probabilidad modelo: **{jugada['Probabilidad modelo']}**")
            if partido["status"] == "FINISHED":
                st.write(f"**{jugada['Resultado']}**")
            st.caption(f"Kalshi: {jugada['Precio ahora']} · {jugada['Disponibilidad']} · Ticker: {jugada.get('Ticker', '—')}")
            if jugada.get("Calibración"):
                st.caption(f"Calibración: {jugada['Calibración']}")
            if jugada.get("capturado_en"):
                st.caption(f"Captura inmutable · {jugada['capturado_en']}")
    if resumen_archivado:
        historial_partido = historial_guardado(partido["id"], predicciones_guardadas, resultados_guardados)
        with st.expander(f"Comparar todas las capturas guardadas ({len(historial_partido)})"):
            st.dataframe(historial_partido, use_container_width=True, hide_index=True)
else:
    if partido["status"] == "FINISHED":
        st.info("No hay una captura previa de este partido en el historial; no se inventa una comparación retrospectiva.")
    else:
        st.info("No encontré mercados modelables de Kalshi para resumir este partido.")

tabla_rendimiento, cantidad_rendimiento = metricas_historial(predicciones_guardadas, resultados_guardados)
with st.expander(f"Aprendizaje y rendimiento histórico ({cantidad_rendimiento} resultados evaluables)"):
    if cantidad_rendimiento:
        st.dataframe(tabla_rendimiento, use_container_width=True, hide_index=True)
        st.caption("El Brier medio mide el error probabilístico; más bajo es mejor. Las filas originales nunca se reescriben.")
    else:
        st.info("El historial empezará a mostrarse después de guardar pronósticos previos y recibir sus resultados.")

if partido["status"] == "FINISHED":
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
            st.dataframe(pd.DataFrame(margen_yes_kalshi), use_container_width=True, hide_index=True)
        else:
            st.info("No hay una apuesta YES disponible ahora que supere el margen mínimo del modelo.")
    with tab_no_kalshi:
        if margen_no_kalshi:
            st.dataframe(pd.DataFrame(margen_no_kalshi), use_container_width=True, hide_index=True)
        else:
            st.info("No hay una apuesta NO disponible ahora que supere el margen mínimo del modelo.")
    if not margen_yes_kalshi and not margen_no_kalshi:
        st.info("Ninguna cotización actual pasa el margen mínimo. Abajo puedes ver los mejores precios actuales y el máximo al que cada selección sí cumpliría el objetivo.")

    with st.expander("Ver los mejores candidatos al precio actual y el precio máximo recomendado"):
        st.caption("Estas filas son diagnósticas: si dicen ‘No tiene margen suficiente’, no cumplen el objetivo al precio mostrado. El precio máximo indica dónde podrían alcanzar la ganancia y el ROI mínimos.")
        yes_cand_tab, no_cand_tab = st.tabs(["Candidatos YES", "Candidatos NO"])
        with yes_cand_tab:
            if candidatos_yes_kalshi:
                st.dataframe(pd.DataFrame(candidatos_yes_kalshi), use_container_width=True, hide_index=True)
            else:
                st.info("No hay cotizaciones YES que pueda comparar para este partido.")
        with no_cand_tab:
            if candidatos_no_kalshi:
                st.dataframe(pd.DataFrame(candidatos_no_kalshi), use_container_width=True, hide_index=True)
            else:
                st.info("No hay cotizaciones NO que pueda comparar para este partido.")

    with st.expander("Ver las 10 opciones más probables aunque no tengan margen"):
        if top_kalshi:
            st.dataframe(pd.DataFrame(top_kalshi), use_container_width=True, hide_index=True)
        else:
            st.info("No encontré tipos y líneas activos de Kalshi en Champions League con estimación suficiente para este partido.")
st.write(f"**Marcador esperado:** {gl:.1f}–{gv:.1f} goles")
st.caption(
    f"Forma reciente (últimos 8 partidos, ajuste limitado): {local} ×{ajuste_forma_local:.2f} · "
    f"{visita} ×{ajuste_forma_visita:.2f}. El histórico completo también pondera más los resultados recientes."
)

with st.expander("Cómo calcula el algoritmo estas probabilidades"):
    st.markdown(
        "**Goles y actualidad:** estima los goles esperados con fuerzas de ataque/defensa de Champions League, ponderadas por antigüedad "
        f"(la mitad del peso cada {HALF_LIFE_DAYS} días) y suavizadas con el promedio de Champions (K=6). Además aplica un ajuste limitado "
        "por los últimos ocho partidos de cada equipo, usando solo encuentros anteriores a la fecha seleccionada. Luego calcula una distribución Poisson para cada equipo "
        "y forma la matriz de marcadores. La probabilidad de cada YES es la suma de las celdas que cumplen esa condición; "
        "la del NO es 1 menos la del YES."
    )
    st.write(f"λ estimada: {local} {gl:.2f} goles · {visita} {gv:.2f} goles")
    st.markdown(
        "**Corners:** Kalshi puede ofrecer este mercado para algunos partidos, pero no hay un historial de corners de Champions conectado a la app. Por eso no se les inventa probabilidad ni margen; revisa las reglas del contrato en Kalshi."
    )
    st.markdown(
        "**Primer y segundo tiempo:** usa la proporción de goles antes del descanso observada en partidos previos ponderados por antigüedad; "
        "si faltan datos, usa 45%/55%. Sigue siendo una estimación aproximada porque no modela jugadas ni alineaciones por periodo. El histórico disponible es de Champions, así que los equipos con pocos partidos europeos se acercan más al promedio de la competición."
    )
    st.markdown(
        "**Margen:** calcula la probabilidad del modelo menos el precio de compra y una tarifa estándar estimada. "
        "Solo recomienda oportunidades con probabilidad ≥50%, ganancia esperada ≥$0.05 y retorno neto esperado ≥10%. "
        "También muestra el precio máximo al que cada selección alcanzaría esos objetivos; si el precio actual es mayor, "
        "la marca como sin margen suficiente. Las tarifas reales pueden variar por mercado y tipo de orden."
    )
    st.caption(
        "Las capturas originales y sus resultados se guardan por separado. Al acumular 30 resultados comparables de una categoría, "
        "la probabilidad futura se calibra con una corrección suavizada; la probabilidad base de cada captura queda intacta."
    )

if partido["status"] != "FINISHED":
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
        st.warning(f"Kalshi no respondió para {len(series_kalshi_con_error)} de {total_series_kalshi} series de Champions League; el listado puede estar incompleto.")
        with st.expander("Ver series de Kalshi que fallaron"):
            st.write(", ".join(series_kalshi_con_error))

    st.caption(
        "Las filas ‘Tipo/línea ofrecido en Champions League; falta abrirlo para este partido’ usan una línea real detectada en otros partidos de Champions. "
        "Sirven como posibilidades del modelo; Kalshi podría no abrir ese contrato para este encuentro."
    )
st.warning(
    "Las probabilidades son estimaciones y no garantías. Para apostar en Kalshi, verifica que el contrato exacto esté disponible "
    "para este partido y revisa las reglas del mercado."
)
st.info("Los hándicaps del resumen describen el resultado binario YES/NO de Kalshi. No significan devolución por empate exacto con la línea; consulta las reglas del contrato y su ticker antes de operar.")
