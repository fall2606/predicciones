"""Contexto deportivo gratuito: noticias públicas para revisar pronósticos."""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Contexto deportivo gratuito", page_icon="🧠", layout="wide")
st.title("🧠 Contexto deportivo · fuentes gratuitas")
st.caption(
    "Consulta noticias públicas recientes sobre equipos y jugadores. No requiere API key ni servicio de IA de pago. "
    "Las noticias sirven para detectar riesgos; no alteran por sí solas las probabilidades ni la calibración."
)

TIMEOUT = 12
MAX_ITEMS_PER_QUERY = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsContextDashboard/1.0)"}

RISK_TERMS = {
    "Lesiones / bajas": ("injury", "injured", "out for", "ruled out", "questionable", "doubtful", "lesión", "lesionado", "baja", "no jugará", "ausente"),
    "Alineación / descanso": ("lineup", "starting", "starter", "rotation", "rested", "resting", "line-up", "alineación", "titular", "descanso"),
    "Sanciones": ("suspended", "suspension", "suspended for", "sancionado", "sanción", "expulsado"),
    "Forma / rendimiento": ("winning streak", "losing streak", "form", "slump", "hot streak", "racha", "rendimiento"),
    "Clima / condiciones": ("weather", "rain", "wind", "heat", "snow", "clima", "lluvia", "viento"),
    "Cambio / controversia": ("trade", "transferred", "manager sacked", "coach fired", "contract", "traspaso", "fichaje", "entrenador destituido"),
}

def parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None

def classify(title: str, summary: str) -> str:
    text = f"{title} {summary}".casefold()
    found = [label for label, terms in RISK_TERMS.items() if any(term in text for term in terms)]
    return ", ".join(found) if found else "Contexto general; verificar relevancia"

@st.cache_data(ttl=900, show_spinner=False)
def search_news(query: str) -> list[dict]:
    """Fetch a public Google News RSS query. Fail closed and preserve source URLs."""
    url = "https://news.google.com/rss/search?q=" + quote_plus(query) + "&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError, ValueError):
        return []
    rows = []
    for item in root.findall("./channel/item")[:MAX_ITEMS_PER_QUERY]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else "Google News RSS"
        published = parse_date((item.findtext("pubDate") or "").strip())
        description = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
        description = re.sub(r"\s+", " ", description).strip()
        if title and link:
            rows.append({
                "Titular": title,
                "Fuente": source,
                "Publicado (UTC)": published.strftime("%Y-%m-%d %H:%M") if published else "Fecha no disponible",
                "Tipo de señal": classify(title, description),
                "Enlace": link,
                "_published": published,
            })
    return rows

with st.form("contexto_deportivo_form"):
    sport = st.selectbox("Deporte", ["Fútbol", "MLB", "NFL", "NBA", "NHL", "Tenis", "Otro"])
    home = st.text_input("Equipo / jugador 1", placeholder="Ej. Real Madrid")
    away = st.text_input("Equipo / jugador 2 (opcional)", placeholder="Ej. Barcelona")
    extra = st.text_input("Jugador clave o tema a comprobar (opcional)", placeholder="Ej. lesión del delantero titular")
    submitted = st.form_submit_button("Buscar contexto reciente", type="primary")

if submitted:
    names = [x.strip() for x in (home, away, extra) if x and x.strip()]
    if not names:
        st.warning("Escribe al menos un equipo o jugador.")
    else:
        all_rows = []
        queries = []
        for name in names:
            queries.append(f'"{name}" {sport} news')
        with st.spinner("Consultando fuentes públicas…"):
            for query in queries:
                all_rows.extend(search_news(query))
        # Deduplicate by URL while retaining the first query's results.
        unique = {}
        for row in all_rows:
            unique.setdefault(row["Enlace"], row)
        rows = list(unique.values())
        rows.sort(key=lambda r: r["_published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        if not rows:
            st.warning("No se pudieron recuperar noticias ahora. Puede ser un fallo temporal del feed; no interpretes esto como ausencia de lesiones o novedades.")
        else:
            st.success(f"{len(rows)} noticias recuperadas. Caché de consulta: hasta 15 minutos.")
            counts = {}
            for row in rows:
                for signal in row["Tipo de señal"].split(", "):
                    counts[signal] = counts.get(signal, 0) + 1
            if counts:
                st.subheader("Señales para revisar")
                st.dataframe(pd.DataFrame([{"Tema": k, "Noticias asociadas": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]), hide_index=True, use_container_width=True)
            st.subheader("Noticias y fuentes")
            for row in rows:
                with st.container(border=True):
                    st.markdown(f"**[{row['Titular']}]({row['Enlace']})**")
                    st.caption(f"{row['Fuente']} · {row['Publicado (UTC)']} UTC")
                    st.write(f"Posible tema: {row['Tipo de señal']}")
            st.info(
                "Control de calidad: las etiquetas son coincidencias de palabras, no una evaluación de veracidad ni un análisis de sentimiento. "
                "Abre la fuente original y confirma la fecha, el jugador afectado y si la noticia aplica al partido. "
                "Una noticia repetida o antigua no debe cambiar la probabilidad."
            )

with st.expander("Cómo usarlo sin contaminar la calibración"):
    st.markdown(
        "- Mantén intacta la probabilidad estadística y la captura original del pronóstico.\n"
        "- Comprueba lesiones, titulares, descanso, sanciones, clima y noticias publicadas después de la última actualización.\n"
        "- No conviertas automáticamente el tono de una noticia en un aumento o reducción de probabilidad.\n"
        "- Registra por separado la fuente y la hora si más adelante quieres medir, con resultados WIN/LOSS, si ese factor mejora el modelo.\n"
        "- Si no hay fuentes fiables o los datos son contradictorios, marca la información como desconocida y no ajustes la predicción."
    )
st.caption("Fuente de búsqueda: Google News RSS público. No es una API deportiva oficial ni garantiza cobertura completa o tiempo real.")
