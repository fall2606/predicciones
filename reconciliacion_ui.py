"""Streamlit rendering for the shared reconciliation report."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reconciliacion_pendientes.csv"


def mostrar_reconciliacion(deporte):
    """Render exactly the same pending definition emitted by auditar_reconciliacion.py."""
    st.subheader("Estado de liquidación y reconciliación")
    if not REPORT.exists():
        st.error("No se puede verificar el contador: falta reconciliacion_pendientes.csv. No se declara cero sin evidencia.")
        return
    try:
        report = pd.read_csv(REPORT, dtype=str).fillna("")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"No se puede leer reconciliacion_pendientes.csv: {exc}")
        return
    required = {"deporte", "prediction_id", "motivo", "cuenta_alerta", "clasificacion"}
    if not required.issubset(report.columns):
        st.error("El informe de reconciliación tiene un esquema antiguo/incompleto. Ejecuta el workflow de reconciliación.")
        return

    current = report[report["deporte"].astype(str).eq(deporte)].copy()
    pending = current[current["cuenta_alerta"].astype(str).str.upper().eq("SI")].copy()
    historic = current[current["clasificacion"].astype(str).eq("HISTORICO_SIN_CONTRATO")].copy()
    historic_nonmatch = current[current["clasificacion"].astype(str).eq("HISTORICO_NO_PARTIDO")].copy()
    resolved = current[current["clasificacion"].astype(str).eq("LIQUIDADA")]
    st.caption(
        f"{deporte}: {len(current)} predicciones auditadas · "
        f"{len(resolved)} con resultado terminal registrado · "
        f"{len(historic)} históricos sin contrato de Kalshi · "
        f"{len(historic_nonmatch)} históricos sin partido asociado."
    )

    if pending.empty:
        st.success("ALERTA: 0 predicciones vencidas sin resultado registrado según el informe de reconciliación.")
    else:
        st.error(f"ALERTA: {len(pending)} predicciones de partidos terminados siguen sin resultado registrado.")
        counts = (
            pending.groupby("motivo", dropna=False)
            .size().reset_index(name="cantidad")
            .sort_values(["cantidad", "motivo"], ascending=[False, True])
        )
        st.markdown("**Desglose por motivo**")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        columns = [
            c for c in ["prediction_id", "match_id", "local", "visita", "bloque", "ticker",
                        "estado_partido", "estado_kalshi", "resultado_kalshi", "motivo",
                        "horas_desde_inicio"] if c in pending.columns
        ]
        st.markdown("**Todas las predicciones que mantienen la alerta**")
        st.dataframe(pending[columns], use_container_width=True, hide_index=True)

    if not historic.empty:
        with st.expander(f"Histórico sin contrato de Kalshi ({len(historic)}; conservado, no cuenta como pendiente activo)", expanded=False):
            cols = [c for c in ["prediction_id", "match_id", "fecha", "local", "visita", "bloque", "ticker", "motivo"] if c in historic.columns]
            st.dataframe(historic[cols].head(100), use_container_width=True, hide_index=True)
            if len(historic) > 100:
                st.caption(f"Se muestran 100 de {len(historic)} filas; el CSV de reconciliación conserva todas.")

    if not historic_nonmatch.empty:
        with st.expander(f"Histórico sin partido asociado ({len(historic_nonmatch)}; por ejemplo, mercados de temporada)", expanded=False):
            cols = [c for c in ["prediction_id", "match_id", "fecha", "local", "visita", "bloque", "ticker", "motivo"] if c in historic_nonmatch.columns]
            st.dataframe(historic_nonmatch[cols].head(100), use_container_width=True, hide_index=True)
            if len(historic_nonmatch) > 100:
                st.caption(f"Se muestran 100 de {len(historic_nonmatch)} filas; el CSV de reconciliación conserva todas.")


def mostrar_actualizacion_archivos(archivos):
    """Mostrar frescura de los CSV leídos directamente por la página, sin caché."""
    zona = ZoneInfo("America/Chicago")
    marcas = []
    for etiqueta, ruta in archivos:
        ruta = Path(ruta)
        try:
            stat = ruta.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, zona)
            data_time = None
            if ruta.suffix.lower() == ".csv" and stat.st_size:
                frame = pd.read_csv(ruta, dtype=str, usecols=lambda col: col in {"resuelto_en", "capturado_en"})
                candidatos = []
                for columna in ("resuelto_en", "capturado_en"):
                    if columna in frame.columns:
                        fechas = pd.to_datetime(frame[columna], utc=True, errors="coerce").dropna()
                        if not fechas.empty:
                            candidatos.append(fechas.max())
                if candidatos:
                    data_time = max(candidatos).to_pydatetime().astimezone(zona)
            marcas.append((etiqueta, ruta.name, mtime, data_time))
        except (OSError, ValueError, pd.errors.ParserError):
            marcas.append((etiqueta, ruta.name, None, None))
    disponibles = [x for x in marcas if x[2] is not None]
    if not disponibles:
        st.warning("No se encontraron los CSV locales de predicciones/resultados: " + ", ".join(nombre for _, nombre, _, _ in marcas))
        return
    ultima_resolucion = max((x[3] for x in marcas if x[3] is not None), default=None)
    texto_resolucion = ultima_resolucion.strftime("%Y-%m-%d %H:%M:%S %Z") if ultima_resolucion else "sin resolución registrada"
    detalles = " · ".join(
        f"{nombre}: archivo {mtime.strftime('%H:%M:%S')}, última fila resuelta/capturada {data_time.strftime('%Y-%m-%d %H:%M:%S %Z') if data_time else 'sin fecha de resolución/captura'}"
        for _, nombre, mtime, data_time in marcas
    )
    st.caption(f"Datos leídos directamente del disco local, sin caché de CSV. Última resolución/captura registrada: {texto_resolucion}. {detalles}")
