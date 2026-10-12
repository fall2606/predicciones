import streamlit as st
from app_metadata import mostrar_metadata

st.set_page_config(page_title="Centro de pronósticos deportivos", page_icon="🎯", layout="centered")
st.title("🎯 Centro de pronósticos deportivos")
mostrar_metadata()
st.write("Elige un deporte y después un partido. La tarjeta distingue el cálculo estadístico de una recomendación respaldada por resultados reales.")
st.warning("Si no hay muestra suficiente, ventaja demostrada o precio de compra válido, la aplicación debe decirlo y no recomendar una apuesta.")

deportes = {
    "LaLiga": ("pages/LaLiga.py", "🇪🇸"),
    "Champions League": ("pages/Champions.py", "🏆"),
    "MLB": ("pages/MLB.py", "⚾"),
    "NFL": ("pages/NFL.py", "🏈"),
    "Premier League": ("pages/Premier_League.py", "⚽"),
}
seleccion = st.selectbox("Deporte", list(deportes), key="deporte_principal")
st.button("Elegir partido", type="primary", on_click=lambda: st.switch_page(deportes[seleccion][0]))
st.divider()
st.page_link("pages/Calidad_del_modelo.py", label="Auditoría · calidad y calibración del modelo", icon="📊")
st.page_link("pages/Contexto_deportivo.py", label="Contexto deportivo", icon="🧠")
st.caption("Regla: si faltan datos verificables, se muestra SIN DATOS; si el modelo no supera el precio de mercado, NO APOSTAR.")
