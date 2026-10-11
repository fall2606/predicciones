"""NFL page: per-game forecasts, original snapshots, calibration and settled results."""
from pathlib import Path
import pandas as pd
import streamlit as st
from reconciliacion_ui import mostrar_actualizacion_archivos

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "predicciones_nfl.csv"
RES = ROOT / "resultados_nfl.csv"
CURRENT = ROOT / "nfl_probabilidades_actuales.csv"
EXCLUDED = ROOT / "nfl_predicciones_excluidas.csv"
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

excluded_df = read_csv(EXCLUDED)
excluded_ids = set(excluded_df["prediction_id"].astype(str)) if not excluded_df.empty and "prediction_id" in excluded_df else set()
if not current.empty and "prediction_id" in current and not pred.empty and "prediction_id" in pred:
    result_fields = [field for field in ("prediction_id", "estado") if field in pred.columns]
    result_lookup = pred[result_fields].drop_duplicates("prediction_id", keep="last")
    current = current.merge(result_lookup, on="prediction_id", how="left", suffixes=("", "_historial"))
    if "estado_historial" in current:
        current["estado"] = current["estado_historial"].where(
            current["estado_historial"].astype(str).str.strip().ne(""), current.get("estado", "")
        )
        current.drop(columns=["estado_historial"], inplace=True)
valid_current = current[~current["prediction_id"].astype(str).isin(excluded_ids)].copy() if not current.empty and "prediction_id" in current else pd.DataFrame()
mostrar_actualizacion_archivos([
    ("Predicciones NFL", PRED), ("Resultados NFL", RES),
    ("Probabilidades actuales NFL", CURRENT), ("Capturas NFL excluidas", EXCLUDED),
])
if pred.empty:
    st.info("Todavía no hay capturas NFL guardadas. El proceso automático guardará solo mercados que pueda modelar con datos suficientes.")
else:
    pred["Estado de captura"] = pred["prediction_id"].apply(
        lambda pid: "EXCLUIDA: captura posterior al inicio real" if str(pid) in excluded_ids else "Válida"
    )
    valid_pred = pred[~pred["prediction_id"].astype(str).isin(excluded_ids)].copy()
    if valid_pred.empty:
        st.warning(f"Hay {len(excluded_ids)} capturas históricas excluidas por haberse registrado después del inicio real. Se conservan en el historial, pero no cuentan para resultados ni calibración.")
    else:
        for field in ("probabilidad_modelo", "probabilidad_calibrada", "precio_captura", "edge"):
            if field in valid_pred:
                valid_pred[field + "_num"] = pd.to_numeric(valid_pred[field], errors="coerce")
        state = valid_pred.get("estado", pd.Series("", index=valid_pred.index)).astype(str).str.upper()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capturas válidas", len(valid_pred))
        c2.metric("WIN", int((state == "WIN").sum()))
        c3.metric("LOSS", int((state == "LOSS").sum()))
        c4.metric("Pendientes", int((~state.isin(["WIN", "LOSS", "ANULADA"])).sum()))
        st.subheader("Selecciones por partido")
        games = sorted(valid_current["match_id"].dropna().astype(str).unique()) if not valid_current.empty and "match_id" in valid_current else []
        if not games:
            st.info("No hay mercados prepartido válidos para recomendar en este momento.")
        for game_id in games:
            group = valid_current[valid_current["match_id"].astype(str) == game_id].copy()
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
    with st.expander("Historial completo de predicciones NFL (incluye capturas excluidas)", expanded=False):
        st.dataframe(pred, use_container_width=True, hide_index=True)
if not current.empty:
    with st.expander("Mercados modelados actuales", expanded=False):
        st.dataframe(current, use_container_width=True, hide_index=True)
st.caption("La calibración exige al menos 30 resultados WIN/LOSS comparables. Los mercados no modelables se omiten en lugar de recibir una probabilidad inventada.")
