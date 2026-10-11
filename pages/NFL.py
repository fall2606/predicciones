from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import nfl_engine as nfl
from reconciliacion_ui import mostrar_actualizacion_archivos, mostrar_reconciliacion

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
TZ_LOCAL = ZoneInfo("America/Chicago")
TERMINAL = {"WIN", "LOSS", "ANULADA"}
FINAL_MATCH_STATES = {
    "FINAL", "FINISHED", "COMPLETED", "STATUS_FINAL", "STATUS_FINAL_OT", "STATUS_FINAL_2OT",
    "CANCELED", "CANCELLED", "STATUS_CANCELED", "STATUS_CANCELLED",
}


st.set_page_config(page_title="NFL | Predicciones FALL", page_icon="🏈", layout="wide")
st.title("🏈 NFL · Predicciones de Kalshi")
st.caption("Mercados reales de Kalshi · modelo NFL propio · cuotas y líneas capturadas antes del inicio")


@st.cache_data(ttl=180, show_spinner=False)
def cargar_calendario_nfl():
    return nfl.load_upcoming_games(days=21)


def leer_csv(path, columns):
    try:
        rows = nfl.read_csv_tolerant(path, expected_columns=columns)
        return pd.DataFrame(rows, columns=list(dict.fromkeys(list(columns) + [c for r in rows for c in r.keys()])))
    except Exception as exc:
        st.warning(f"No se pudo leer {Path(path).name}; la página continúa sin esos datos. Detalle: {exc}")
        return pd.DataFrame(columns=columns)


def commit_corto():
    for key in ("GITHUB_SHA", "COMMIT_SHA", "STREAMLIT_GIT_COMMIT"):
        value = os.getenv(key)
        if value:
            return value[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=3, check=True,
        )
        return result.stdout.strip() or "no disponible"
    except Exception:
        return "no disponible"


def parse_date(value):
    dt = nfl.parse_dt(value)
    return dt.astimezone(ET) if dt else None


def estado_resultado(row):
    state = str(row.get("estado") or "").upper()
    if state in TERMINAL:
        return state
    match_state = str(row.get("estado_partido") or "").upper()
    if match_state in FINAL_MATCH_STATES or match_state.startswith("STATUS_FINAL"):
        return "SIN LIQUIDAR"
    return "PENDIENTE"


def color_resultado(value):
    colors = {
        "PENDIENTE": "background-color: #777777; color: white; font-weight: bold",
        "WIN": "background-color: #218838; color: white; font-weight: bold",
        "LOSS": "background-color: #c82333; color: white; font-weight: bold",
        "ANULADA": "background-color: #6f42c1; color: white; font-weight: bold",
        "SIN LIQUIDAR": "background-color: #b21f2d; color: white; font-weight: bold",
    }
    return colors.get(str(value), "")


def centavos(value):
    try:
        return f"{float(value) * 100:.0f}¢"
    except (TypeError, ValueError):
        return "Sin precio en Kalshi"


def puntos_porcentuales(value):
    try:
        return f"{float(value) * 100:+.1f} pp"
    except (TypeError, ValueError):
        return "—"


def probabilidad(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def categoria_legible(value):
    return {
        "winner": "Ganador del partido",
        "spread": "Hándicap de puntos",
        "total": "Total de puntos",
        "team_total": "Total de puntos por equipo",
    }.get(str(value), str(value or "Mercado NFL"))


def mostrar_tarjeta(row):
    state = estado_resultado(row)
    with st.container(border=True):
        st.markdown(f"**{row.get('descripcion') or row.get('mercado') or 'Predicción NFL'}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad del modelo", probabilidad(row.get("probabilidad_modelo")))
        c2.metric("Precio de Kalshi", centavos(row.get("precio_captura")))
        c3.metric("Ventaja", puntos_porcentuales(row.get("edge")))
        st.markdown(f"**Resultado:** {state}")
        st.caption(f"{categoria_legible(row.get('categoria'))} · Ticker: {row.get('ticker') or '—'}")


def tabla_bloque(frame, titulo):
    st.subheader(titulo)
    if frame.empty:
        st.info("No hay predicciones en este bloque para el partido seleccionado.")
        return
    cols = [
        ("descripcion", "Apuesta"), ("categoria", "Categoría"),
        ("probabilidad_modelo", "Probabilidad"), ("precio_captura", "Precio Kalshi"),
        ("edge", "Ventaja (pp)"), ("estado", "Resultado"), ("ticker", "Ticker"),
    ]
    data = pd.DataFrame()
    for source, label in cols:
        if source not in frame.columns:
            continue
        if source == "descripcion":
            data[label] = frame[source].astype(str)
        elif source == "categoria":
            data[label] = frame[source].map(categoria_legible)
        elif source == "probabilidad_modelo":
            data[label] = frame[source].map(probabilidad)
        elif source == "precio_captura":
            data[label] = frame[source].map(centavos)
        elif source == "edge":
            data[label] = frame[source].map(puntos_porcentuales)
        else:
            data[label] = frame[source].astype(str)
    if "Resultado" in data.columns:
        st.dataframe(data.style.map(color_resultado, subset=["Resultado"]), use_container_width=True, hide_index=True)
    else:
        st.dataframe(data, use_container_width=True, hide_index=True)


if st.button("🔄 Actualizar datos ahora", type="primary"):
    st.cache_data.clear()
    st.rerun()

predicciones = leer_csv(ROOT / "predicciones_nfl.csv", nfl.PRED_COLUMNS)
resultados = leer_csv(ROOT / "resultados_nfl.csv", nfl.RESULT_COLUMNS)
cobertura = leer_csv(ROOT / "nfl_cobertura_por_partido.csv", nfl.COVERAGE_COLUMNS)
catalogo = leer_csv(ROOT / "nfl_catalogo_actual.csv", [
    "event_ticker", "ticker", "series_ticker", "event_title", "market_title", "market_status",
    "yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars", "modelable", "motivo",
])
descartados = leer_csv(ROOT / "nfl_candidatos_descartados.csv", [
    "match_id", "local", "visita", "event_ticker", "ticker", "categoria", "motivo",
])

if not resultados.empty and {"prediction_id", "estado"}.issubset(resultados.columns):
    resultados["resuelto_en"] = resultados.get("resuelto_en", "").astype(str)
    resultados = resultados.sort_values("resuelto_en").drop_duplicates("prediction_id", keep="last")
    resultados = resultados[["prediction_id", "estado", "marcador", "fuente", "motivo", "resuelto_en"]]
    predicciones = predicciones.drop(columns=[c for c in ("estado", "marcador", "fuente", "motivo", "resuelto_en") if c in predicciones.columns], errors="ignore")
    predicciones = predicciones.merge(resultados, on="prediction_id", how="left")
else:
    for col in ("estado", "marcador", "fuente", "motivo", "resuelto_en"):
        if col not in predicciones.columns:
            predicciones[col] = ""

if not predicciones.empty:
    predicciones["estado"] = predicciones.get("estado", "").fillna("").astype(str).str.upper()
    predicciones["edge_num"] = pd.to_numeric(predicciones.get("edge", ""), errors="coerce")
    predicciones["prob_num"] = pd.to_numeric(predicciones.get("probabilidad_modelo", ""), errors="coerce")
else:
    predicciones["edge_num"] = pd.Series(dtype=float)
    predicciones["prob_num"] = pd.Series(dtype=float)

recon = pd.DataFrame()
recon_path = ROOT / "reconciliacion_pendientes.csv"
if recon_path.exists():
    try:
        recon = pd.read_csv(recon_path, dtype=str).fillna("")
        if "deporte" in recon.columns:
            recon = recon[recon["deporte"].astype(str).str.upper().eq("NFL")]
    except (OSError, ValueError, pd.errors.ParserError):
        recon = pd.DataFrame()
match_status = {}
if not recon.empty and {"match_id", "estado_partido"}.issubset(recon.columns):
    match_status = recon.drop_duplicates("match_id").set_index("match_id")["estado_partido"].to_dict()
if not predicciones.empty and "match_id" in predicciones.columns:
    predicciones["estado_partido"] = predicciones["match_id"].map(match_status).fillna("")
    predicciones["resultado_mostrar"] = predicciones.apply(estado_resultado, axis=1)
else:
    predicciones["estado_partido"] = ""
    predicciones["resultado_mostrar"] = pd.Series(dtype=str)

try:
    calendario = cargar_calendario_nfl()
except Exception as exc:
    calendario = []
    st.warning(f"No se pudo cargar el calendario NFL ahora mismo. Las predicciones guardadas siguen disponibles. Detalle: {exc}")

ahora = datetime.now(timezone.utc)
opciones = {}
for game in calendario:
    dt = parse_date(game.get("date"))
    if not dt:
        continue
    if dt >= ahora.astimezone(ET) - timedelta(days=14):
        match_id = game.get("match_id") or f"NFL:{dt.date()}:{game.get('away')}@{game.get('home')}"
        opciones[match_id] = {
            "match_id": match_id, "fecha": dt.isoformat(), "local": game.get("home", ""),
            "visita": game.get("away", ""), "status": game.get("status", ""),
        }
if not predicciones.empty:
    for match_id, group in predicciones.groupby("match_id", dropna=False):
        if not match_id:
            continue
        dt = parse_date(group.iloc[0].get("hora_inicio") or group.iloc[0].get("fecha"))
        if dt and dt >= ahora.astimezone(ET) - timedelta(days=14):
            opciones.setdefault(match_id, {
                "match_id": match_id, "fecha": dt.isoformat(),
                "local": group.iloc[0].get("local", ""), "visita": group.iloc[0].get("visita", ""),
                "status": group.iloc[0].get("estado_partido", ""),
            })

if opciones:
    option_ids = sorted(opciones, key=lambda key: opciones[key]["fecha"])
    labels = {
        key: f"{parse_date(opciones[key]['fecha']).strftime('%a %b %d, %I:%M %p ET')} · {opciones[key]['visita']} @ {opciones[key]['local']}"
        for key in option_ids
    }
    selected_id = st.selectbox("Partido", option_ids, format_func=lambda key: labels[key])
    game_info = opciones[selected_id]
    st.markdown(f"### {game_info['visita']} @ {game_info['local']}")
    dt = parse_date(game_info["fecha"])
    if dt:
        st.caption(f"Inicio previsto: {dt.strftime('%A, %d %b %Y · %I:%M %p ET')} · Match ID: {selected_id}")
    rows = predicciones[predicciones.get("match_id", pd.Series(dtype=str)).astype(str).eq(selected_id)].copy() if not predicciones.empty else pd.DataFrame()
    if not rows.empty:
        rows = rows.sort_values(["bloque", "edge_num"], ascending=[True, False])
        singles = rows[rows["bloque"].astype(str).eq("SENCILLAS")].head(5)
        st.subheader("SENCILLAS")
        if singles.empty:
            st.info("Ninguna sencilla pasó los filtros configurados para este partido.")
        else:
            cols = st.columns(min(3, len(singles)))
            for index, (_, row) in enumerate(singles.iterrows()):
                with cols[index % len(cols)]:
                    mostrar_tarjeta(row)
        tabla_bloque(rows[rows["bloque"].astype(str).eq("ALTA_PROBABILIDAD")].head(5), "5 de alta probabilidad para combos")
        tabla_bloque(rows[rows["bloque"].astype(str).eq("A_TENER_EN_CUENTA")].head(10), "10 a tener en cuenta")
        historic = rows[~rows["estado"].astype(str).str.upper().isin(TERMINAL)]
        if not historic.empty:
            with st.expander("Predicciones aún no liquidadas de este partido", expanded=False):
                st.dataframe(historic[["descripcion", "ticker", "lado", "bloque", "estado", "estado_partido", "motivo"]], use_container_width=True, hide_index=True)
    else:
        cov = cobertura[cobertura.get("match_id", pd.Series(dtype=str)).astype(str).eq(selected_id)] if not cobertura.empty else pd.DataFrame()
        if not cov.empty:
            item = cov.iloc[0]
            messages = [
                ("Sencillas", item.get("motivo_sencillas", "")),
                ("5 de alta probabilidad", item.get("motivo_alta_probabilidad", "")),
                ("10 a tener en cuenta", item.get("motivo_a_tener_en_cuenta", "")),
            ]
            st.info("El partido fue revisado, pero no hay apuestas que superen los filtros.")
            for title, reason in messages:
                if reason:
                    st.write(f"**{title}:** {reason}")
        else:
            st.info("Todavía no hay predicciones guardadas para este partido. El pipeline debe capturar el catálogo de Kalshi antes del inicio.")
else:
    st.info("No hay partidos NFL próximos en el calendario disponible. Si existen predicciones históricas guardadas, se mostrarán en la sección de histórico.")

st.divider()
st.subheader("Histórico")
if predicciones.empty:
    st.info("Todavía no hay predicciones NFL guardadas.")
else:
    historic = predicciones.copy()
    historic["fecha_dt"] = pd.to_datetime(historic.get("fecha", ""), utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=14)
    historic = historic[(historic["fecha_dt"] < cutoff) | historic.get("ticker", "").astype(str).eq("")]
    with st.expander(f"Ver histórico conservado ({len(historic)} filas)", expanded=False):
        if historic.empty:
            st.caption("Aún no hay predicciones anteriores a los últimos 14 días.")
        else:
            cols = [c for c in ["fecha", "visita", "local", "descripcion", "ticker", "lado", "bloque", "probabilidad_modelo", "precio_captura", "edge", "resultado_mostrar", "marcador", "fuente"] if c in historic.columns]
            hist_view = historic[cols].rename(columns={"resultado_mostrar": "Resultado"})
            if "Resultado" in hist_view.columns:
                st.dataframe(hist_view.style.map(color_resultado, subset=["Resultado"]), use_container_width=True, hide_index=True)
            else:
                st.dataframe(hist_view, use_container_width=True, hide_index=True)

sin_precio = pd.DataFrame()
if not catalogo.empty and {"modelable", "yes_ask_dollars", "no_ask_dollars"}.issubset(catalogo.columns):
    yes_price = pd.to_numeric(catalogo["yes_ask_dollars"], errors="coerce")
    no_price = pd.to_numeric(catalogo["no_ask_dollars"], errors="coerce")
    sin_precio = catalogo[catalogo["modelable"].astype(str).eq("SI") & yes_price.isna() & no_price.isna()]
if not sin_precio.empty:
    with st.expander(f"Mercados calculables sin precio en Kalshi ({len(sin_precio)})", expanded=False):
        st.dataframe(sin_precio[["event_title", "market_title", "ticker"]], use_container_width=True, hide_index=True)

if not descartados.empty:
    with st.expander(f"Mercados descartados (no se generan; {len(descartados)})", expanded=False):
        st.dataframe(descartados.head(200), use_container_width=True, hide_index=True)
        if len(descartados) > 200:
            st.caption(f"Se muestran 200 de {len(descartados)} descartes; el CSV conserva el informe completo.")

mostrar_reconciliacion("NFL")
mostrar_actualizacion_archivos([
    ("Predicciones NFL", ROOT / "predicciones_nfl.csv"),
    ("Resultados NFL", ROOT / "resultados_nfl.csv"),
    ("Catálogo Kalshi NFL", ROOT / "nfl_catalogo_actual.csv"),
    ("Cobertura NFL", ROOT / "nfl_cobertura_por_partido.csv"),
    ("Reconciliación", ROOT / "reconciliacion_pendientes.csv"),
])

st.caption(
    f"Última lectura de la página: {datetime.now(TZ_LOCAL).strftime('%Y-%m-%d %H:%M:%S %Z')} · "
    f"Commit de la aplicación: {commit_corto()} · Calendario cacheado durante 180 segundos."
)
