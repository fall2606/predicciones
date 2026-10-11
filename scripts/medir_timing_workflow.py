"""Measure GitHub Actions schedule-to-start and queue delay using run metadata."""
import os
from datetime import datetime, timezone, timedelta
import requests


def dt(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def main():
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("No hay token para leer los metadatos del run.")
    response = requests.get(
        f"https://api.github.com/repos/{repository}/actions/runs/{run_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=8,
    )
    response.raise_for_status()
    run = response.json()
    created = dt(run["created_at"])
    started = dt(run.get("run_started_at") or run["created_at"])
    slot = created.replace(minute=(created.minute // 5) * 5, second=0, microsecond=0)
    scheduler_lag = (created - slot).total_seconds()
    queue_delay = (started - created).total_seconds()
    print(f"TIMING ACTIONS: evento={run.get('event')} · run={run_id}")
    print(f"TIMING ACTIONS: cron_slot_estimado_UTC={slot.isoformat()} · run_created_at_UTC={created.isoformat()} · run_started_at_UTC={started.isoformat()}")
    print(f"TIMING ACTIONS: desfase_creacion_vs_slot_estimado_seg={scheduler_lag:.1f} · cola_hasta_inicio_seg={queue_delay:.1f}")
    print("TIMING ACTIONS: el slot es estimado a partir de la creación del run; no sustituye un timestamp de disparo programado suministrado por GitHub.")


if __name__ == "__main__":
    main()
