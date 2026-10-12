from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="Calidad del modelo", page_icon="📊", layout="wide")
st.title("📊 ¿Qué tan buenos son los pronósticos?")
st.write("Esta página compara las probabilidades guardadas con los resultados reales. No usa tablas: muestra las cifras que necesitas para saber si un modelo merece confianza.")
st.warning("Importante: estas cifras son descriptivas. Para demostrar que un modelo predice bien hace falta un backtest cronológico fuera de muestra y comparar el modelo con el mercado. Una tasa de acierto alta por sí sola no basta.")


def leer_csv(nombre):
    ruta = ROOT / nombre
    if not ruta.exists() or ruta.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(ruta, dtype=str, keep_default_na=False, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def probabilidad(columna):
    original = columna.astype(str).str.strip()
    es_porcentaje = original.str.contains("%", regex=False)
    valores = pd.to_numeric(original.str.replace("%", "", regex=False), errors="coerce")
    # Convert only values explicitly marked as percentages. Invalid fractions > 1 stay invalid.
    return valores.where(~es_porcentaje, valores / 100)


def crear_intervalos(p, y):
    filas = []
    indices = np.minimum((p * 10).astype(int), 9)
    for indice in range(10):
        mascara = indices == indice
        inicio = indice / 10
        if mascara.any():
            filas.append({"Intervalo estimado": f"{inicio:.0%}–{inicio + 0.1:.0%}", "Casos": int(mascara.sum()), "Probabilidad media": float(p[mascara].mean()), "Acierto real": float(y[mascara].mean())})
    return filas


def evaluar_archivos(nombre, archivo_pred, archivo_res, claves_prob, ticker_col="ticker"):
    pred = leer_csv(archivo_pred)
    res = leer_csv(archivo_res)
    if pred.empty or res.empty or "prediction_id" not in pred or "prediction_id" not in res:
        return {"nombre": nombre, "n": 0, "total": len(pred), "sin_resultado": len(pred), "estado": "Aún no evaluable"}
    res = res.sort_values("resuelto_en") if "resuelto_en" in res else res
    res = res.drop_duplicates("prediction_id", keep="last")
    unido = pred.merge(res[["prediction_id", "estado"]], on="prediction_id", how="left", suffixes=("", "_resultado"))
    # Si la captura también tiene una columna "estado", esa describe la captura,
    # no necesariamente el resultado final. Para evaluar se prioriza el CSV de resultados.
    columna_estado_resultado = "estado_resultado" if "estado_resultado" in unido.columns else "estado"
    estado = unido[columna_estado_resultado].astype(str).str.upper().str.strip()
    liquidado = unido[estado.isin(["WIN", "LOSS"])].copy()
    p = pd.Series(np.nan, index=liquidado.index, dtype=float)
    for clave in claves_prob:
        if clave in liquidado.columns:
            candidato = probabilidad(liquidado[clave])
            p = p.fillna(candidato)
    y = liquidado[columna_estado_resultado].astype(str).str.upper().eq("WIN").astype(float)
    valido = p.notna() & np.isfinite(p) & p.between(0, 1)
    p = p[valido]
    y = y[valido]
    n = len(p)
    if n == 0:
        return {"nombre": nombre, "n": 0, "total": len(pred), "sin_resultado": int((~estado.isin(["WIN", "LOSS", "ANULADA"])).sum()), "intervalos": [], "estado": "Sin probabilidades y resultados compatibles"}
    tasa = float(y.mean())
    brier = float(((p - y) ** 2).mean())
    p_segura = p.clip(1e-15, 1 - 1e-15)
    log_loss = float((-(y * np.log(p_segura) + (1 - y) * np.log(1 - p_segura))).mean())
    accuracy = float(((p >= 0.5) == (y == 1)).mean())
    media_p = float(p.mean())
    baseline = tasa * (1 - tasa)
    real_ticker = int((liquidado.get(ticker_col, pd.Series("", index=liquidado.index)).astype(str).str.strip().replace({"—": "", "-": ""}) != "").sum()) if ticker_col in liquidado.columns else n
    if n < 30:
        estado_modelo = "Muestra insuficiente"
    elif brier >= baseline:
        estado_modelo = "No supera la referencia simple"
    else:
        estado_modelo = "Señal preliminar favorable; falta backtest temporal"
    return {
        "nombre": nombre, "n": n, "total": len(pred),
        "sin_resultado": int((~estado.isin(["WIN", "LOSS", "ANULADA"])).sum()),
        "accuracy": accuracy, "brier": brier, "log_loss": log_loss, "baseline": baseline,
        "media_p": media_p, "tasa": tasa, "real_ticker": real_ticker,
        "intervalos": crear_intervalos(p, y), "estado": estado_modelo,
    }


def evaluar_premier_league():
    df = leer_csv("predicciones.csv")
    if df.empty or not {"p", "gano"}.issubset(df.columns):
        return {"nombre": "Premier League", "n": 0, "total": len(df), "sin_resultado": len(df), "estado": "Aún no evaluable"}
    p = pd.to_numeric(df["p"], errors="coerce")
    y = pd.to_numeric(df["gano"], errors="coerce")
    valid = p.notna() & y.isin([0, 1]) & p.between(0, 1)
    p, y = p[valid], y[valid]
    n = len(p)
    if not n:
        return {"nombre": "Premier League", "n": 0, "total": len(df), "sin_resultado": len(df), "estado": "Sin resultados compatibles"}
    tasa = float(y.mean())
    brier = float(((p-y)**2).mean())
    return {
        "nombre": "Premier League", "n": n, "total": len(df), "sin_resultado": int(pd.to_numeric(df["gano"], errors="coerce").isna().sum()),
        "accuracy": float(((p>=0.5)==(y==1)).mean()), "brier": brier, "log_loss": float((-(y*np.log(p.clip(1e-15, 1-1e-15))+(1-y)*np.log(1-p.clip(1e-15, 1-1e-15)))).mean()), "baseline": tasa*(1-tasa),
        "media_p": float(p.mean()), "tasa": tasa, "real_ticker": n, "intervalos": crear_intervalos(p, y),
        "estado": "No supera la referencia simple" if n >= 30 and brier >= tasa*(1-tasa) else ("Muestra insuficiente" if n < 30 else "Señal preliminar favorable; falta backtest temporal"),
    }


resultados = [
    evaluar_premier_league(),
    evaluar_archivos("LaLiga", "predicciones_laliga.csv", "resultados_laliga.csv", ("probabilidad_calibrada", "probabilidad_base")),
    evaluar_archivos("Champions League", "predicciones_champions.csv", "resultados_champions.csv", ("probabilidad_calibrada", "probabilidad_base")),
    evaluar_archivos("MLB", "predicciones_mlb.csv", "resultados_mlb.csv", ("probabilidad_calibrada", "probabilidad_modelo")),
    evaluar_archivos("NFL", "predicciones_nfl.csv", "resultados_nfl.csv", ("probabilidad_calibrada", "probabilidad_modelo")),
]

for fila in resultados:
    with st.container(border=True):
        st.subheader(fila["nombre"])
        st.caption(fila["estado"])
        if fila["n"] == 0:
            st.info(f"No hay muestra evaluable en los archivos disponibles. Capturas en archivo: {fila['total']}.")
            continue
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resultados evaluados", fila["n"])
        c2.metric("Acierto", f"{fila['accuracy']:.1%}")
        c3.metric("Brier (menor es mejor)", f"{fila['brier']:.3f}")
        c4.metric("Log-loss (menor es mejor)", f"{fila.get('log_loss', float('nan')):.3f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("Probabilidad media", f"{fila['media_p']:.1%}")
        c5.metric("Acierto real", f"{fila['tasa']:.1%}")
        c6.metric("Referencia Brier", f"{fila['baseline']:.3f}")
        st.caption(f"Capturas totales: {fila['total']} · con ticker en las liquidaciones: {fila.get('real_ticker', fila['n'])} · sin resultado terminal asociado: {fila['sin_resultado']}. El contador puede incluir partidos futuros.")
        intervalos = fila.get("intervalos", [])
        if intervalos:
            st.caption("Calibración por intervalos: compara la probabilidad media con el acierto real; grupos pequeños son inestables.")
            st.dataframe(pd.DataFrame(intervalos).style.format({"Probabilidad media": "{:.1%}", "Acierto real": "{:.1%}"}), hide_index=True, use_container_width=True)
        if fila["brier"] >= fila["baseline"]:
            st.warning("En esta muestra, el Brier no mejora una referencia que siempre predice la tasa media de acierto. No tratar los porcentajes como validados.")
        elif fila["n"] < 30:
            st.warning("La muestra es demasiado pequeña para confiar en la diferencia.")
        else:
            st.info("Resultado preliminar: falta comprobarlo con separación temporal y contra precios de mercado.")

st.divider()
st.markdown("**Cómo leerlo:** el Brier mide el error de las probabilidades, no solo si gana la selección. Se compara aquí con una referencia sencilla de la misma muestra; esto no es un backtest independiente. Las predicciones relacionadas de un mismo partido tampoco son observaciones independientes.")
