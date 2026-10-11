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
    if run.get("event") != "schedule":
        print(f"TIMING ACTIONS: no aplica a evento={run.get('event')}; se mide solo en ejecuciones programadas.")
        return
    created = dt(run["created_at"])
    slot = created.replace(minute=(created.minute // 5) * 5, second=0, microsecond=0)
    jobs_response = requests.get(
        f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/jobs",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=8,
    )
    jobs_response.raise_for_status()
    jobs = jobs_response.json().get("jobs") or []
    starts = [dt(job["started_at"]) for job in jobs if job.get("started_at")]
    actual_start = min(starts) if starts else dt(run.get("run_started_at") or run["created_at"])
    scheduler_lag = (created - slot).total_seconds()
    queue_delay = (actual_start - created).total_seconds()
    print(f"TIMING ACTIONS: evento={run.get('event')} · run={run_id}")
    print(f"TIMING ACTIONS: slot_cron_estimado_UTC={slot.isoformat()} · run_created_at_UTC={created.isoformat()} · primer_job_started_at_UTC={actual_start.isoformat()}")
    print(f"TIMING ACTIONS: desfase_creacion_vs_slot_estimado_seg={scheduler_lag:.1f} · cola_hasta_inicio_real_seg={queue_delay:.1f}")
    print("TIMING ACTIONS: el slot se infiere redondeando hacia abajo al múltiplo de 5 minutos; si GitHub omite un disparo no puede detectarse solo con este run.")


if __name__ == "__main__":
    main()
