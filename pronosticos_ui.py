"""Presentación sencilla de pronósticos: separa estimaciones del modelo de apuestas reales."""
import re
import streamlit as st


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
    """Muestra tarjetas legibles, nunca etiqueta una estimación como apuesta."""
    if not filas:
        st.info("Todavía no hay pronósticos disponibles para mostrar.")
        return
    st.caption(
        f"**Pronóstico del modelo ({origen}).** Son estimaciones basadas en datos; "
        "no son apuestas de Kalshi ni una garantía de resultado. Una probabilidad más alta "
        "significa que el modelo considera ese resultado más posible, no seguro."
    )
    for i, fila in enumerate(filas):
        partido = str(fila.get("Partido") or fila.get("Juego") or f"Partido {i + 1}")
        fecha = str(fila.get("Fecha (Chicago)") or fila.get("Fecha (UTC)") or fila.get("Fecha") or "")
        estado = str(fila.get("Estado del cálculo") or "")
        with st.expander(f"{partido}" + (f" · {fecha}" if fecha else ""), expanded=(i == 0)):
            if estado and estado.upper() not in {"OK", "PROBABILIDAD BASE"}:
                st.warning("No se pudo calcular una predicción completa para este partido. " + estado.replace("NO DISPONIBLE:", "Motivo técnico:"))
                continue
            marcador = fila.get("Goles esperados") or fila.get("Marcador esperado") or fila.get("Marcador estimado (local-visita)")
            if marcador:
                st.write(f"**Marcador estimado:** {marcador}")
            favorito = fila.get("Favorito del modelo")
            if favorito:
                st.write(f"**Equipo que el modelo ve con más posibilidades:** {favorito}")
            probabilidades = []
            for clave, valor in fila.items():
                if clave in {"Partido", "Juego", "Fecha (Chicago)", "Fecha (UTC)", "Fecha", "Estado del cálculo",
                             "Goles esperados", "Marcador esperado", "Marcador estimado (local-visita)",
                             "Favorito", "Favorito del modelo", "Resultado 1X2 más probable"}:
                    continue
                p = _es_probabilidad(valor)
                if p is not None:
                    etiqueta = clave.replace("Gana local", "Gana el equipo local").replace(
                        "Gana visitante", "Gana el equipo visitante"
                    ).replace("Más de", "Que haya más de")
                    probabilidades.append((etiqueta, p))
            principal = fila.get("Resultado 1X2 más probable")
            if principal:
                st.write(f"**Resultado más probable entre victoria local, empate o victoria visitante:** {principal}")
            if probabilidades:
                cols = st.columns(min(3, len(probabilidades)))
                for j, (etiqueta, prob) in enumerate(probabilidades):
                    with cols[j % len(cols)]:
                        st.metric(etiqueta, f"{prob:.0%}")
            if not probabilidades and not principal and not favorito:
                st.info("El modelo no devolvió probabilidades utilizables para este partido.")
            st.info("**Cómo leerlo:** estos porcentajes describen lo que estima el modelo. No indican cuánto ganarías, no comparan cuotas y no significan que debas apostar.")



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
        st.info("No hay selecciones que mostrar con los filtros actuales.")
        return
    for fila in filas:
        nombre = _valor_fila(
            fila,
            ("Jugada sencilla", "Apuesta", "Pronóstico", "Mercado", "Mercado Kalshi", "mercado", "seleccion", "Tipo", "Categoría resumen"),
            "Selección sin nombre",
        )
        categoria = _valor_fila(fila, ("Tipo", "Categoría resumen", "categoria", "bloque"), "")
        if str(categoria).casefold() == str(nombre).casefold():
            categoria = ""
        prob = _porcentaje_legible(_valor_fila(
            fila,
            ("Prob. recalibrada", "Probabilidad recalculada", "Probabilidad del modelo",
             "Probabilidad modelo", "Prob. modelo", "Prob. original", "Probabilidad estimada",
             "probabilidad_calibrada", "probabilidad_modelo"),
        ))
        precio_raw = _valor_fila(
            fila,
            ("Precio de compra", "Precio Kalshi", "Precio actual", "Precio captura", "Precio", "Precio ahora", "precio"),
        )
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
        edge = _ventaja_legible(_valor_fila(
            fila, ("Ventaja neta", "Ventaja neta recalculada", "Ventaja (edge)", "edge", "EV neto por contrato"),
        ))
        ticker = _valor_fila(fila, ("Ticker", "ticker"), "")
        calibracion = _valor_fila(fila, ("Calibración", "calibracion"), "")
        resultado = str(_valor_fila(fila, ("Resultado", "estado"), "PENDIENTE")).upper()
        with st.container(border=True):
            st.markdown(f"**{nombre}**")
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
            c2.metric("Precio de compra", precio_texto)
            c3.metric("Ventaja estimada", edge)
            if calibracion:
                st.caption(f"Validación: {calibracion}")
            if "WIN" == resultado or resultado.startswith("WIN "):
                st.success("Resultado: WIN")
            elif "LOSS" == resultado or resultado.startswith("LOSS "):
                st.error("Resultado: LOSS")
            elif "ANULADA" in resultado or "VOID" in resultado:
                st.info("Resultado: ANULADA")
            elif "SIN DATOS" in resultado or "NO EVALUABLE" in resultado:
                st.warning("Resultado: requiere revisión; no se inventa.")
            else:
                st.caption("Resultado: pendiente de confirmar")
            if not ticker:
                st.warning("No hay ticker confirmado: no es un contrato ejecutable de Kalshi.")
            else:
                st.caption(f"Contrato Kalshi: {ticker}")
            if precio == "—":
                st.caption("Sin precio de compra confirmado, no se calcula una ventaja ejecutable.")
