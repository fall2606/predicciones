import streamlit as st

st.set_page_config(page_title="Centro de pronósticos deportivos", page_icon="🎯", layout="wide")
st.title("🎯 Centro de pronósticos deportivos")
st.write("Elige un deporte. Primero verás lo que estima el modelo; los contratos reales de Kalshi aparecen aparte. Un porcentaje no garantiza un resultado.")
st.warning("La calidad histórica todavía no está validada por igual en todos los deportes. Si no hay datos suficientes o una ventaja comprobable, no tomes el pronóstico como recomendación de apuesta.")

st.subheader("Elige qué quieres analizar")
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("### ⚽ Premier League")
        st.write("Probabilidades de partido y goles estimados.")
        st.page_link("pages/Premier_League.py", label="Ver pronósticos Premier League", icon="⚽")
    with st.container(border=True):
        st.markdown("### 🇪🇸 LaLiga")
        st.write("Pronósticos del modelo, contratos y resultados.")
        st.page_link("pages/LaLiga.py", label="Ver LaLiga", icon="🇪🇸")
    with st.container(border=True):
        st.markdown("### 🏆 Champions League")
        st.write("Pronósticos, contratos y resultados archivados.")
        st.page_link("pages/Champions.py", label="Ver Champions League", icon="🏆")
with c2:
    with st.container(border=True):
        st.markdown("### ⚾ MLB")
        st.write("Ganador, carreras y mercados de entradas cuando hay datos.")
        st.page_link("pages/MLB.py", label="Ver MLB", icon="⚾")
    with st.container(border=True):
        st.markdown("### 🏈 NFL")
        st.write("Marcador proyectado y mercados de partido modelables.")
        st.page_link("pages/NFL.py", label="Ver NFL", icon="🏈")
    with st.container(border=True):
        st.markdown("### 🧠 Contexto deportivo")
        st.write("Noticias para revisar manualmente; no ajustan automáticamente las probabilidades.")
        st.page_link("pages/Contexto_deportivo.py", label="Revisar contexto y noticias", icon="🧠")

st.divider()
st.caption("Regla de lectura: 1) pronóstico del modelo, 2) motivo y datos usados, 3) contrato real y precio, 4) resultado WIN/LOSS. Si falta precio, ticker o evidencia, no es una apuesta ejecutable.")
