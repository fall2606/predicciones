"""Deterministic, auditable reasoning over public sports-news items (no paid AI API)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping

SIGNALS = {
    "injury": ("injury", "injured", "ruled out", "questionable", "doubtful", "lesion", "lesionado", "lesión", "baja", "no jugará", "ausente"),
    "lineup": ("lineup", "line-up", "starting", "starter", "rotation", "rested", "resting", "alineación", "titular", "descanso"),
    "suspension": ("suspended", "suspension", "sancionado", "sanción", "expulsado"),
    "form": ("winning streak", "losing streak", "slump", "hot streak", "racha", "rendimiento", "form"),
    "weather": ("weather", "rain", "wind", "heat", "snow", "clima", "lluvia", "viento"),
    "roster_change": ("trade", "transferred", "manager sacked", "coach fired", "traspaso", "fichaje", "entrenador destituido"),
}
SIGNAL_LABELS = {
    "injury": "Lesiones/bajas", "lineup": "Alineación/descanso",
    "suspension": "Sanciones", "form": "Forma/rendimiento",
    "weather": "Clima/condiciones", "roster_change": "Cambios de plantilla/cuerpo técnico",
}
NEGATIVE_CUES = ("out", "injury", "injured", "ruled out", "doubtful", "questionable", "suspended", "loss", "losing", "lesión", "lesionado", "baja", "sancionado")
POSITIVE_CUES = ("returns", "available", "cleared to play", "back in training", "win streak", "winning", "regresa", "disponible", "vuelve")
UNKNOWN_DATE = "Fecha no disponible"


def _parse_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def analyze_news(items: Iterable[Mapping], entities: Iterable[str] = (), now=None) -> dict:
    """Explain observed signals and data gaps; deliberately never alters forecast probabilities."""
    now = _parse_datetime(now) or datetime.now(timezone.utc)
    entities = [str(x).casefold().strip() for x in entities if str(x).strip()]
    analyzed = []
    signal_counts = {}
    fresh_count = 0
    old_count = 0
    for item in items:
        title = str(item.get("Titular") or item.get("title") or "").strip()
        summary = str(item.get("Resumen") or item.get("summary") or "").strip()
        source = str(item.get("Fuente") or item.get("source") or "Fuente no identificada").strip()
        url = str(item.get("Enlace") or item.get("url") or "").strip()
        published = _parse_datetime(item.get("_published") or item.get("published"))
        text = f"{title} {summary}".casefold()
        matched_entities = [e for e in entities if e in text]
        signals = [key for key, terms in SIGNALS.items() if any(term in text for term in terms)]
        age_hours = (now - published).total_seconds() / 3600 if published else None
        is_fresh = age_hours is not None and 0 <= age_hours <= 72
        if is_fresh:
            fresh_count += 1
        elif age_hours is not None:
            old_count += 1
        # Never infer positive/negative sporting impact from keywords alone.
        polarity = "mixta/ambigua"
        if any(term in text for term in NEGATIVE_CUES) and not any(term in text for term in POSITIVE_CUES):
            polarity = "posible riesgo; confirmar"
        elif any(term in text for term in POSITIVE_CUES) and not any(term in text for term in NEGATIVE_CUES):
            polarity = "posible noticia favorable; confirmar"
        relevance = "alta: menciona equipo/jugador buscado" if matched_entities else "no confirmada: verificar que se refiere al equipo/jugador"
        if not signals:
            signals = ["general"]
        for signal in signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        analyzed.append({
            "title": title, "source": source, "url": url,
            "published": published.isoformat() if published else UNKNOWN_DATE,
            "age_hours": round(age_hours, 1) if age_hours is not None else None,
            "fresh": is_fresh, "entities": matched_entities,
            "signals": signals, "polarity": polarity, "relevance": relevance,
        })
    analyzed.sort(key=lambda x: x["age_hours"] if x["age_hours"] is not None else float("inf"))
    if not analyzed:
        status = "INSUFFICIENT_DATA"
        summary = "No se recuperaron noticias; no puede concluirse que no existan lesiones o novedades."
    elif fresh_count == 0:
        status = "STALE_OR_UNDATED"
        summary = "No hay noticias fechadas dentro de las últimas 72 horas. No uses estas noticias para ajustar el pronóstico."
    elif not any(x["entities"] for x in analyzed):
        status = "RELEVANCE_UNCONFIRMED"
        summary = "Hay noticias recientes, pero ninguna confirma claramente la entidad buscada en el titular o resumen."
    else:
        status = "REVIEW_REQUIRED"
        summary = "Hay señales recientes relacionadas; verifica las fuentes originales y el impacto específico antes del partido."
    return {
        "status": status, "summary": summary, "total": len(analyzed),
        "fresh_count": fresh_count, "old_count": old_count,
        "signal_counts": {SIGNAL_LABELS.get(k, "Contexto general"): v for k, v in signal_counts.items()},
        "items": analyzed,
        "probability_adjustment": None,
        "decision": "NO_AUTOMATIC_ADJUSTMENT",
    }
