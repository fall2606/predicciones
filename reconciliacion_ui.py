"""Streamlit rendering for the shared reconciliation report."""
from pathlib import Path

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
    resolved = current[current["clasificacion"].astype(str).eq("LIQUIDADA")]
    st.caption(
        f"{deporte}: {len(current)} predicciones auditadas · "
        f"{len(resolved)} con resultado terminal registrado · "
        f"{len(historic)} históricos sin contrato de Kalshi · "
        f"plazo de liquidación Kalshi configurable en el workflow."
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
            st.dataframe(historic[cols], use_container_width=True, hide_index=True)
