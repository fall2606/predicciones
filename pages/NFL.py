"""NFL page: per-game forecasts, original snapshots, calibration and settled results."""
from pathlib import Path
import pandas as pd
import streamlit as st
from reconciliacion_ui import mostrar_actualizacion_archivos

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "predicciones_nfl.csv"
RES = ROOT / "resultados_nfl.csv"
CURRENT = ROOT / "nfl_probabilidades_actuales.csv"
st.set_page_config(page_title="Predicciones NFL", page_icon="🏈", layout="wide")
st.title("🏈 NFL · predicciones por partido")
st.caption("Modelo específico de fútbol americano con resultados y precios de Kalshi. La probabilidad original se conserva; la calibración solo usa WIN/LOSS resueltos.")
if st.button("Actualizar datos NFL ahora", key="refresh_nfl"):
    st.cache_data.clear()
    st.rerun()

def read_csv(path):
    try:
        return pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame()
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()

pred = read_csv(PRED)
res = read_csv(RES)
current = read_csv(CURRENT)
if not pred.empty and not res.empty and "prediction_id" in pred and "prediction_id" in res:
    results = res.sort_values("resuelto_en").drop_duplicates("prediction_id", keep="last")
    pred = pred.merge(results, on="prediction_id", how="left", suffixes=("", "_resultado"))
    for field in ("estado", "resultado_kalshi", "marcador", "resuelto_en"):
        alt = field + "_resultado"
        if alt in pred:
            if field in pred:
                pred[field] = pred[alt].where(pred[alt].astype(str).str.strip().ne(""), pred[field])
            else:
                pred[field] = pred[alt]
            pred.drop(columns=[alt], inplace=True)

mostrar_actualizacion_archivos([("Predicciones NFL", PRED), ("Resultados NFL", RES), ("Probabilidades actuales NFL", CURRENT)])
if pred.empty:
    st.info("Todavía no hay capturas NFL guardadas. El pipeline debe ejecutar nfl_engine.py cuando haya mercados NFL abiertos y datos suficientes.")
else:
    for field in ("probabilidad_modelo", "probabilidad_calibrada", "precio_captura", "edge"):
        if field in pred:
            pred[field + "_num"] = pd.to_numeric(pred[field], errors="coerce")
    state = pred.get("estado", pd.Series("", index=pred.index)).astype(str).str.upper()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capturas", len(pred))
    c2.metric("WIN", int((state == "WIN").sum()))
    c3.metric("LOSS", int((state == "LOSS").sum()))
    c4.metric("Pendientes", int((~state.isin(["WIN", "LOSS", "ANULADA"])).sum()))
    st.subheader("Selecciones por partido")
    games = sorted(pred["match_id"].dropna().astype(str).unique()) if "match_id" in pred else []
    for game_id in games:
        group = pred[pred["match_id"].astype(str) == game_id].copy()
        game_name = f"{group.iloc[0].get('local','')} vs {group.iloc[0].get('visita','')}"
        with st.expander(game_name + " · " + str(group.iloc[0].get("fecha",""))[:10], expanded=True):
            for block, label, limit in (("SENCILLAS","5 posibles sencillas",5),("ALTA_PROBABILIDAD","5 selecciones para combos",5),("A_TENER_EN_CUENTA","10 a tener en cuenta",10),("DETALLE","Otros mercados modelados",1000)):
                subset = group[group.get("bloque", pd.Series("", index=group.index)).eq(block)].copy()
                if subset.empty:
                    if block != "DETALLE":
                        st.caption(f"{label}: no hay suficientes selecciones válidas en este partido.")
                    continue
                st.markdown(f"**{label}**")
                display = pd.DataFrame({
                    "Mercado": subset.get("mercado", ""),
                    "Lado": subset.get("lado", ""),
                    "Prob. original": (pd.to_numeric(subset.get("probabilidad_modelo"), errors="coerce")*100).round(1).astype(str)+"%",
                    "Prob. recalibrada": (pd.to_numeric(subset.get("probabilidad_calibrada", subset.get("probabilidad_modelo")), errors="coerce")*100).round(1).astype(str)+"%",
                    "Precio captura": subset.get("precio_captura", ""),
                    "Ventaja": (pd.to_numeric(subset.get("edge"), errors="coerce")*100).round(1).astype(str)+"%",
                    "Resultado": subset.get("estado", pd.Series("PENDIENTE", index=subset.index)).replace("", "PENDIENTE"),
                    "Ticker": subset.get("ticker", ""),
                })
                st.dataframe(display.head(limit), use_container_width=True, hide_index=True)
    with st.expander("Historial completo de predicciones NFL", expanded=False):
        st.dataframe(pred, use_container_width=True, hide_index=True)
if not current.empty:
    with st.expander("Mercados modelados actuales", expanded=False):
        st.dataframe(current, use_container_width=True, hide_index=True)
st.caption("La calibración exige al menos 30 resultados WIN/LOSS comparables. Los mercados no modelables se omiten en lugar de recibir una probabilidad inventada.")
