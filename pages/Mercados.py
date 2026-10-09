import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Todos los mercados", layout="wide")
st.title("📋 Todos los mercados por partido")

if not os.path.exists("predicciones.csv"):
    st.info("Todavía no hay predicciones guardadas. Corre el workflow en la pestaña Actions de GitHub.")
    st.stop()

df = pd.read_csv("predicciones.csv")
df["partido"] = df["fecha"] + " · " + df["local"] + " vs " + df["visita"]
lista = sorted(df["partido"].unique())
elegido = st.selectbox("Partido", lista)
d = df[df["partido"] == elegido].copy()

d["Prob %"] = (d["p"] * 100).round(1)
d["Cuota justa"] = (1 / d["p"].clip(lower=0.001)).round(2)

def grupo(m):
    if m in ("Gana local", "Empate", "Gana visita"):
        return "Ganador"
    if "gana por más" in m:
        return "Hándicap"
    if m.startswith("Más de"):
        return "Total de goles"
    if m.startswith("Local más") or m.startswith("Visita más"):
        return "Totales por equipo"
    if m == "Ambos marcan":
        return "Ambos equipos marcan"
    if m.startswith("Marcador"):
        return "Marcador exacto (los 8 más probables)"
    return "Primer equipo en anotar"

d["Grupo"] = d["mercado"].map(grupo)
orden = ["Ganador", "Hándicap", "Total de goles", "Totales por equipo",
         "Ambos equipos marcan", "Marcador exacto (los 8 más probables)",
         "Primer equipo en anotar"]

st.caption("Prob % = precio justo en centavos. Si en Kalshi el precio de Yes es menor, el modelo ve valor (value).")
for gr in orden:
    sub = d[d["Grupo"] == gr]
    if gr.startswith("Marcador"):
        sub = sub.sort_values("p", ascending=False).head(8)
    if len(sub):
        st.subheader(gr)
        st.dataframe(sub[["mercado", "Prob %", "Cuota justa"]],
                     use_container_width=True, hide_index=True)
