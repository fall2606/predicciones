"""Build/version metadata shown in every Streamlit page without claiming stale data."""
from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

ROOT = Path(__file__).resolve().parent
TZ = ZoneInfo("America/Chicago")
DATA_FILES = (
    "predicciones_laliga.csv", "resultados_laliga.csv",
    "predicciones_champions.csv", "resultados_champions.csv",
    "predicciones_mlb.csv", "resultados_mlb.csv",
    "predicciones_nfl.csv", "resultados_nfl.csv",
    "mlb_probabilidades_actuales.csv", "nfl_probabilidades_actuales.csv",
    "reconciliacion_pendientes.csv",
)


def current_commit():
    for name in ("APP_COMMIT_SHA", "GITHUB_SHA", "COMMIT_SHA"):
        value = str(os.getenv(name, "")).strip()
        if value:
            return value[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=2, check=True,
        )
        value = result.stdout.strip()
        return value or "no disponible"
    except (OSError, subprocess.SubprocessError):
        return "no disponible en el entorno"


def latest_data_time():
    latest_row_time = None
    latest_file_time = None
    timestamp_fields = ("capturado_en", "resuelto_en", "actualizado_en", "updated_at", "fecha_actualizacion")
    for filename in DATA_FILES:
        path = ROOT / filename
        if not path.is_file():
            continue
        try:
            file_time = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            latest_file_time = max(latest_file_time, file_time) if latest_file_time else file_time
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    for field in timestamp_fields:
                        raw = str(row.get(field, "") or "").strip()
                        if not raw:
                            continue
                        try:
                            stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                            if stamp.tzinfo is None:
                                stamp = stamp.replace(tzinfo=timezone.utc)
                            stamp = stamp.astimezone(timezone.utc)
                            latest_row_time = max(latest_row_time, stamp) if latest_row_time else stamp
                        except (TypeError, ValueError, OverflowError):
                            continue
        except (OSError, csv.Error, UnicodeError):
            continue
    # Prefer timestamps recorded with the actual data. File mtime is only a fallback,
    # because a deploy/checkout time is not necessarily a data-refresh time.
    latest = latest_row_time or latest_file_time
    return latest.astimezone(TZ).strftime("%Y-%m-%d %H:%M %Z") if latest else "sin fecha disponible"


def mostrar_metadata():
    st.caption(f"Commit: {current_commit()} · Última actualización de archivos/datos: {latest_data_time()}")
