"""Presentación sencilla de pronósticos: separa estimaciones del modelo de apuestas reales."""
import os
import re
import streamlit as st
from prediction_core import anchor_probability_to_market, probability_fraction, evaluate_market_recommendation


def _es_probabilidad(valor):
    texto = str(valor or "").strip()
    match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*%", texto)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ".")) / 100
    except ValueError:
        return None


def mostrar_pronosticos_sencillos(filas, deporte="deporte", origen="Modelo estadístico"):
    """Permite analizar un partido a la vez y distingue pronóstico de apuesta."""
    if not filas:
        st.info("Todavía no hay pronósticos disponibles para mostrar.")
        return

    # Un selector evita una pared de tarjetas con todos los partidos a la vez.
    filas = list(filas)
    opciones = []
    por_etiqueta = {}
    for i, fila in enumerate(filas):
        partido = str(fila.get("Partido") or fila.get("Juego") or f"Partido {i + 1}")
        fecha = str(fila.get("Fecha (Chicago)") or fila.get("Fecha (UTC)") or fila.get("Fecha") or "")
        etiqueta = f"{fecha} · {partido}" if fecha else partido
        if etiqueta in por_etiqueta:
            etiqueta = f"{etiqueta} · {i + 1}"
        opciones.append(etiqueta)
        por_etiqueta[etiqueta] = fila

    clave = re.sub(r"[^a-zA-Z0-9_]", "_", str(deporte).lower())
    seleccionado = st.selectbox(
        "Elige un partido para ver su pronóstico",
        opciones,
        key=f"selector_pronostico_{clave}",
    )
    fila = por_etiqueta[seleccionado]
    st.caption(
        f"**Estimación del modelo · {origen}.** Es un pronóstico estadístico, no una apuesta "
        "de Kalshi ni una garantía. La calidad solo se considera demostrada si supera una "
        "evaluación histórica fuera de muestra."
    )

    partido = str(fila.get("Partido") or fila.get("Juego") or "Partido seleccionado")
    estado = str(fila.get("Estado del cálculo") or "")
    if estado and estado.upper() not in {"OK", "PROBABILIDAD BASE"}:
        st.warning(
            "No se pudo calcular una predicción completa para este partido. "
            + estado.replace("NO DISPONIBLE:", "Motivo técnico:")
        )
        return

    with st.container(border=True):
        st.subheader(partido)
        fecha = str(fila.get("Fecha (Chicago)") or fila.get("Fecha (UTC)") or fila.get("Fecha") or "")
        if fecha:
            st.caption(f"Fecha del partido: {fecha}")
        marcador = fila.get("Goles esperados") or fila.get("Marcador esperado") or fila.get("Marcador estimado (local-visita)")
        if marcador:
            st.write(f"**Marcador estimado:** {marcador}")
        favorito = fila.get("Favorito del modelo")
        if favorito:
            st.write(f"**Equipo con más posibilidades según el modelo:** {favorito}")

        probabilidades = []
        ignorar = {
            "Partido", "Juego", "Fecha (Chicago)", "Fecha (UTC)", "Fecha",
            "Estado del cálculo", "Goles esperados", "Marcador esperado",
            "Marcador estimado (local-visita)", "Favorito", "Favorito del modelo",
            "Resultado 1X2 más probable",
        }
        for clave_prob, valor in fila.items():
            if clave_prob in ignorar:
                continue
            p = _es_probabilidad(valor)
            if p is not None:
                etiqueta = clave_prob.replace("Gana local", "Gana el equipo local").replace(
                    "Gana visitante", "Gana el equipo visitante"
                ).replace("Más de", "Que haya más de")
                probabilidades.append((etiqueta, p))
        principal = fila.get("Resultado 1X2 más probable")
        if principal:
            st.write(f"**Resultado más probable entre victoria local, empate o victoria visitante:** {principal}")
        if probabilidades:
            for etiqueta, prob in probabilidades:
                st.metric(etiqueta, f"{prob:.0%}")
        if not probabilidades and not principal and not favorito:
            st.info("El modelo no devolvió probabilidades utilizables para este partido.")

        fundamento = next((
            str(fila[k]).strip() for k in
            ("Fundamento", "Datos usados", "Razón del modelo", "Explicación", "Nota del modelo")
            if fila.get(k) and str(fila[k]).strip() not in {"—", "-", "nan", "None"}
        ), "")
        st.markdown("**¿Qué respalda esta predicción?**")
        if fundamento:
            st.write(fundamento)
        else:
            st.write(
                "Esta pantalla solo contiene la estimación estadística y los datos que se muestran arriba. "
                "No presenta una comprobación automática de lesiones, alineaciones, noticias ni clima, "
                "y no demuestra por sí sola que exista ventaja frente al precio de una apuesta."
            )
        st.caption(
            "Probabilidad final anclada a Kalshi: no disponible en esta tarjeta independiente, porque no contiene "
            "un precio de compra verificable. No se sustituye ese dato por la probabilidad del modelo."
        )
        st.info(
            "**Importante:** estos porcentajes indican lo que estima el modelo. No dicen cuánto ganarías "
            "ni significan que debas apostar. Para valorar una apuesta hacen falta precio de compra actual, "
            "costes, mercado confirmado y evidencia histórica suficiente."
        )


def _valor_fila(fila, claves, defecto="—"):
    for clave in claves:
        valor = fila.get(clave)
        if valor is not None and str(valor).strip() not in {"", "nan", "None", "—", "-"}:
            return valor
    return defecto


def _porcentaje_legible(valor):
    texto = str(valor or "").strip()
    if not texto or texto in {"—", "-", "None", "nan"}:
        return "—"
    if "%" in texto:
        return texto
    try:
        numero = float(texto)
        if 0 <= numero <= 1:
            return f"{numero:.0%}"
        if 0 <= numero <= 100:
            return f"{numero:.1f}%"
    except (TypeError, ValueError):
        pass
    return texto


def _ventaja_legible(valor):
    texto = str(valor or "").strip()
    if not texto or texto in {"—", "-", "None", "nan"}:
        return "No calculable"
    try:
        numero = float(texto.replace("%", "").replace("pp", "").strip())
        if abs(numero) > 1:
            numero /= 100
        return f"{numero * 100:+.1f} puntos porcentuales"
    except (TypeError, ValueError):
        return texto


def mostrar_tarjetas_mercado(filas, titulo=None, limite=10):
    """Render concise, readable cards for market picks instead of dense tables."""
    if hasattr(filas, "to_dict"):
        filas = filas.to_dict("records")
    filas = list(filas or [])[:limite]
    if titulo:
        st.caption(titulo)
    if not filas:
        st.info("Hoy no hay recomendaciones: no hay candidatos con ventaja y datos suficientes para pasar los filtros actuales.")
        return
    gates = []
    for fila in filas:
        estado_actual = str(_valor_fila(fila, ("Resultado", "estado"), "PENDIENTE")).upper()
        if estado_actual in {"WIN", "LOSS", "ANULADA"}:
            gates.append(None)
        else:
            gates.append(evaluate_market_recommendation(fila))
    gates_activos = [gate for gate in gates if gate is not None]
    elegibles = sum(1 for gate in gates_activos if gate.get("eligible"))
    if gates_activos and not elegibles:
        razones = list(dict.fromkeys(gate["reason"] for gate in gates_activos if gate.get("reason")))
        detalle = "; ".join(razones[:3]) or "faltan resultados liquidados comparables, ventaja validada o precio ask"
        st.info(f"Hoy no hay recomendaciones. Motivo(s): {detalle}.")
    for fila, gate in zip(filas, gates):
        nombre = _valor_fila(
            fila,
            ("Jugada sencilla", "Apuesta", "Pronóstico", "Mercado", "Mercado Kalshi", "mercado", "seleccion", "Tipo", "Categoría resumen"),
            "Selección sin nombre",
        )
        categoria = _valor_fila(fila, ("Tipo", "Categoría resumen", "categoria", "bloque"), "")
        ticker = _valor_fila(fila, ("Ticker", "ticker"), "")
        if str(categoria).casefold() == str(nombre).casefold():
            categoria = ""
        model_raw = _valor_fila(
            fila,
            ("Probabilidad del modelo", "Probabilidad modelo", "Prob. modelo",
             "Prob. recalibrada", "Probabilidad recalculada", "Prob. original",
             "Probabilidad estimada", "probabilidad_calibrada", "probabilidad_modelo"),
        )
        prob = _porcentaje_legible(model_raw)
        precio_raw = _valor_fila(
            fila,
            ("Precio de compra", "Precio Kalshi", "Precio actual", "Precio captura", "Precio", "Precio ahora", "precio", "precio_captura"),
        )
        final_raw = _valor_fila(fila, ("Probabilidad final", "probabilidad_final"), "")
        final_num = None if gate is not None else (probability_fraction(final_raw) if final_raw else None)
        if final_num is None:
            final_num = anchor_probability_to_market(model_raw, precio_raw, category=categoria, ticker=ticker)
        final_text = _porcentaje_legible(final_num) if final_num is not None else "Sin precio válido"
        precio_texto = str(precio_raw)
        if not any(token in precio_texto for token in ("$", "¢", "%")):
            try:
                precio_num = float(precio_texto)
                if 0 < precio_num < 1:
                    precio_texto = f"${precio_num:.2f} ({precio_num * 100:.0f}¢)"
                elif 1 < precio_num <= 100:
                    precio_texto = f"{precio_num:.0f}¢"
            except (TypeError, ValueError):
                pass
        precio_num = probability_fraction(precio_raw)
        if final_num is not None and precio_num is not None:
            try:
                coste = max(0.0, float(os.getenv("MERCADOS_BUFFER_COSTES", "0.01")))
            except (TypeError, ValueError, OverflowError):
                coste = 0.01
            edge = _ventaja_legible(final_num - precio_num - coste)
        else:
            edge = "No calculable sin precio válido"
        calibracion = _valor_fila(fila, ("Calibración", "calibracion"), "")
        resultado = str(_valor_fila(fila, ("Resultado", "estado"), "PENDIENTE")).upper()
        with st.container(border=True):
            st.markdown(f"**{nombre}**")
            if gate is None:
                veredicto = "NO APOSTAR · mercado ya liquidado"
                confianza = "No aplica"
            elif gate.get("eligible"):
                veredicto = "APOSTAR"
                mejora = gate.get("brier_market", 0) - gate.get("brier_model", 0)
                confianza = "Alta" if gate.get("n", 0) >= 200 and mejora >= 0.03 else "Media"
            elif "no supera" in gate.get("reason", "") or "ventaja neta no positiva" in gate.get("reason", ""):
                veredicto = "NO APOSTAR"
                confianza = "No validada" if "muestra insuficiente" in gate.get("reason", "") else "Baja"
            else:
                veredicto = "SIN DATOS"
                confianza = "No validada"
            if gate is not None and gate.get("eligible"):
                st.success(f"Veredicto: {veredicto}")
            else:
                st.warning(f"Veredicto: {veredicto} · Sin recomendación: {gate.get('reason', '') if gate else 'el resultado ya está cerrado'}")
            lado = _valor_fila(fila, ("Lado", "Lado Kalshi", "lado", "seleccion"), "")
            if lado and str(lado).upper() in {"YES", "NO"}:
                st.caption(f"{categoria + ' · ' if categoria else ''}Lado del contrato: {str(lado).upper()}")
            elif categoria:
                st.caption(str(categoria))
            fundamento = _valor_fila(fila, ("Fundamento", "Dato base", "Marcador proyectado", "Nota"), "")
            if fundamento:
                st.write(f"**Dato que apoya la estimación:** {fundamento}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilidad del modelo", prob)
            c2.metric("Probabilidad final", final_text)
            c3.metric("Precio de compra", precio_texto)
            c4, c5 = st.columns(2)
            c4.metric("Ventaja vs. precio", edge)
            c5.metric("Nivel de confianza", confianza)
            if final_num is None:
                st.caption("No se calcula probabilidad final: falta una cotización de compra válida de Kalshi.")
            if gate is not None:
                st.write(f"**Por qué:** {gate.get('reason', 'no cumple la compuerta de calidad')}")
            if calibracion:
                st.caption(f"Calibración por mercado: {calibracion}")
            if "WIN" == resultado or resultado.startswith("WIN "):
                st.success("Resultado: WIN")
            elif "LOSS" == resultado or resultado.startswith("LOSS "):
                st.error("Resultado: LOSS")
            elif any(token in resultado for token in ("ANULADA", "VOID", "CANCELLED", "POSTPONED")):
                st.info("Resultado: ANULADA")
            elif "SIN DATOS" in resultado or "NO EVALUABLE" in resultado or "SIN LIQUIDAR" in resultado:
                st.warning("Resultado: PENDIENTE · requiere revisión de la fuente de liquidación.")
            else:
                st.caption("Resultado: PENDIENTE")
            if not ticker:
                st.warning("No hay ticker confirmado: no es un contrato ejecutable de Kalshi.")
            else:
                st.caption(f"Contrato Kalshi: {ticker}")
            if precio_raw == "—":
                st.caption("Sin precio de compra confirmado, no se calcula una ventaja ejecutable.")
