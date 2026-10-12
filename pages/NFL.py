"""NFL page: per-game forecasts, original snapshots, calibration and settled results."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
from reconciliacion_ui import mostrar_actualizacion_archivos
from nfl_engine import ESPN, _json as nfl_json, _team_stats, projected_score, _normal_cdf

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "predicciones_nfl.csv"
RES = ROOT / "resultados_nfl.csv"
CURRENT = ROOT / "nfl_probabilidades_actuales.csv"
EXCLUDED = ROOT / "nfl_predicciones_excluidas.csv"
st.set_page_config(page_title="Predicciones NFL", page_icon="🏈", layout="wide")
st.title("🏈 NFL · predicciones por partido")
st.page_link("pages/Contexto_deportivo.py", label="Analizar noticias, lesiones y contexto antes de decidir", icon="🧠")
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

# Modelo de marcador NFL independiente del catálogo de Kalshi.
# Así se pronostican los encuentros aunque aún no existan contratos negociables.
@st.cache_data(ttl=900, show_spinner=False)
def cargar_pronosticos_nfl_independientes():
    ahora = datetime.now(timezone.utc)
    temporada_modelo = ahora.year if ahora.month >= 8 else ahora.year - 1
    stats_modelo = _team_stats()
    data = nfl_json(ESPN, {"dates": str(temporada_modelo), "seasontype": 2, "limit": 1000})
    filas_modelo = []
    for evento in data.get("events", []):
        try:
            comp = (evento.get("competitions") or [{}])[0]
            estado = comp.get("status", {}).get("type", {})
            if estado.get("completed"):
                continue
            fecha = datetime.fromisoformat(str(evento.get("date", "")).replace("Z", "+00:00"))
            if fecha < ahora or fecha > ahora + timedelta(days=14):
                continue
            competidores = comp.get("competitors") or []
            home = next(x for x in competidores if x.get("homeAway") == "home")
            away = next(x for x in competidores if x.get("homeAway") == "away")
            local = home.get("team", {}).get("displayName") or home.get("team", {}).get("name")
            visita = away.get("team", {}).get("displayName") or away.get("team", {}).get("name")
            marcador = projected_score(local, visita, stats_modelo)
            if not marcador:
                filas_modelo.append({"Fecha (UTC)": fecha.strftime("%d/%m %H:%M"), "Partido": f"{local} vs {visita}", "Estado del cálculo": "Sin muestra suficiente"})
                continue
            puntos_local, puntos_visita = marcador
            p_local = _normal_cdf(puntos_local - puntos_visita, 0, 13.5)
            p_visita = 1 - p_local
            p_over = _normal_cdf(puntos_local + puntos_visita, 40.5, 10.5)
            favorito, p_favorito = (local, p_local) if p_local >= p_visita else (visita, p_visita)
            filas_modelo.append({
                "Fecha (UTC)": fecha.strftime("%d/%m %H:%M"),
                "Partido": f"{local} vs {visita}",
                "Marcador esperado": f"{puntos_local:.1f} - {puntos_visita:.1f}",
                "Gana local": f"{p_local:.1%}",
                "Gana visitante": f"{p_visita:.1%}",
                "Más de 40.5 puntos": f"{p_over:.1%}",
                "Favorito del modelo": f"{favorito} ({p_favorito:.1%})",
                "Estado del cálculo": "Probabilidad base",
            })
        except (StopIteration, KeyError, TypeError, ValueError, requests.RequestException):
            continue
    return filas_modelo

st.subheader("Pronósticos NFL independientes de Kalshi")
st.caption("Cálculo estadístico de puntos esperados a partir de resultados recientes de ESPN. Se muestran probabilidades base, sin afirmar que estén calibradas; la ausencia de un mercado de Kalshi no elimina el pronóstico.")
try:
    _filas_nfl_modelo = cargar_pronosticos_nfl_independientes()
    if _filas_nfl_modelo:
        st.dataframe(pd.DataFrame(_filas_nfl_modelo), use_container_width=True, hide_index=True)
    else:
        st.info("ESPN no devolvió partidos futuros modelables en los próximos 14 días.")
except (requests.RequestException, ValueError, KeyError, TypeError) as _exc:
    st.warning(f"No pude calcular el pronóstico NFL independiente ahora: {type(_exc).__name__}. Los mercados e historiales de Kalshi siguen disponibles abajo.")

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
        settled = valid_pred[state.isin(["WIN", "LOSS"])].copy()
        settled_p = pd.to_numeric(settled.get("probabilidad_modelo", pd.Series(dtype=float)), errors="coerce")
        valid_p = settled_p.between(0, 1)
        brier = None
        if valid_p.any():
            outcomes = settled.loc[valid_p, "estado"].astype(str).str.upper().eq("WIN").astype(float)
            brier = float(((settled_p.loc[valid_p] - outcomes) ** 2).mean())
        settled_count = int(state.isin(["WIN", "LOSS"]).sum())
        hit_rate = float((state == "WIN").sum() / settled_count) if settled_count else None
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Capturas válidas", len(valid_pred))
        c2.metric("WIN", int((state == "WIN").sum()))
        c3.metric("LOSS", int((state == "LOSS").sum()))
        c4.metric("Pendientes", int((~state.isin(["WIN", "LOSS", "ANULADA"])).sum()))
        c5.metric("Acierto WIN/LOSS", f"{hit_rate:.1%}" if hit_rate is not None else "—")
        c6.metric("Brier (menor es mejor)", f"{brier:.3f}" if brier is not None else "—")
        st.subheader("Selecciones por partido")
        games = sorted(valid_current["match_id"].dropna().astype(str).unique()) if not valid_current.empty and "match_id" in valid_current else []
        if not games:
            st.info("No hay mercados prepartido válidos para recomendar en este momento.")
        else:
            opciones = {}
            for game_id in games:
                game_rows = valid_current[valid_current["match_id"].astype(str) == game_id]
                first = game_rows.iloc[0]
                local = str(first.get("local", "")).strip() or "Local"
                visita = str(first.get("visita", "")).strip() or "Visitante"
                fecha = str(first.get("fecha", ""))[:16].replace("T", " ")
                opciones[f"{fecha} · {visita} @ {local}"] = game_id
            indice_inicial = 0
            for i, game_id in enumerate(opciones.values()):
                filas = valid_current[valid_current["match_id"].astype(str) == game_id]
                if filas["estado"].astype(str).str.upper().apply(lambda x: x not in {"WIN", "LOSS", "ANULADA"}).any():
                    indice_inicial = i
                    break
            etiqueta_partido = st.selectbox(
                "Elige un partido (solo se muestran sus predicciones)",
                list(opciones.keys()),
                index=indice_inicial,
                key="selector_partido_nfl",
            )
            game_id = opciones[etiqueta_partido]
            group = valid_current[valid_current["match_id"].astype(str) == game_id].copy()
            st.caption("Se muestran únicamente las selecciones del partido elegido. Cambia el partido desde el menú desplegable.")
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
                    "Ventaja neta": (pd.to_numeric(subset.get("edge"), errors="coerce")*100).round(1).astype(str)+"%",
                    "Calibración": subset.get("calibracion", pd.Series("Sin dato", index=subset.index)).replace("", "Sin dato"),
                    "Resultado": subset.get("estado", pd.Series("PENDIENTE", index=subset.index)).replace("", "PENDIENTE"),
                    "Ticker": subset.get("ticker", ""),
                })
                st.dataframe(display.head(limit), use_container_width=True, hide_index=True)
                if block != "DETALLE" and len(subset) < limit:
                    st.caption(f"{len(subset)} de {limit} disponibles; no se completan cupos con selecciones que no superan los filtros del modelo.")
    with st.expander("Historial completo de predicciones NFL (incluye capturas excluidas)", expanded=False):
        st.dataframe(pred, use_container_width=True, hide_index=True)
if not current.empty:
    with st.expander("Mercados modelados actuales", expanded=False):
        st.dataframe(current, use_container_width=True, hide_index=True)
st.caption("La calibración exige al menos 30 resultados WIN/LOSS comparables. Los mercados no modelables se omiten en lugar de recibir una probabilidad inventada.")
