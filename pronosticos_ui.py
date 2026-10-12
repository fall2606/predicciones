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
