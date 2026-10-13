import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from football_match_utils import (
    apply_reconciliation_statuses,
    is_upcoming_match,
    merge_match_sources,
)


class FootballMatchUtilsTests(unittest.TestCase):
    def test_past_scheduled_match_is_not_upcoming(self):
        match = {"id": 1, "status": "SCHEDULED", "utcDate": "2026-10-10T18:00:00Z"}
        now = datetime(2026, 10, 12, 17, 0, tzinfo=timezone.utc)
        self.assertFalse(is_upcoming_match(match, now))

    def test_future_scheduled_match_is_upcoming(self):
        match = {"id": 1, "status": "TIMED", "utcDate": "2026-10-13T18:00:00Z"}
        now = datetime(2026, 10, 12, 17, 0, tzinfo=timezone.utc)
        self.assertTrue(is_upcoming_match(match, now))

    def test_finished_match_never_upcoming_even_if_future_date(self):
        match = {"id": 1, "status": "FINISHED", "utcDate": "2026-10-13T18:00:00Z"}
        self.assertFalse(is_upcoming_match(match))

    def test_stale_scheduled_copy_cannot_overwrite_final(self):
        old = {"id": 1, "status": "FINISHED", "score": {"fullTime": {"home": 2, "away": 0}}}
        stale = {"id": 1, "status": "SCHEDULED", "score": {"fullTime": {"home": None, "away": None}}}
        merged = merge_match_sources([old], [stale])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["status"], "FINISHED")
        self.assertEqual(merged[0]["score"]["fullTime"], {"home": 2, "away": 0})

    def test_espn_final_status_is_never_upcoming(self):
        match = {"id": 8, "status": "STATUS_FINAL", "utcDate": "2026-10-13T18:00:00Z"}
        self.assertFalse(is_upcoming_match(match))

    def test_champions_league_reconciliation_alias_promotes_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reconciliacion_pendientes.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["deporte", "match_id", "estado_partido"])
                writer.writeheader()
                writer.writerow({"deporte": "UEFA Champions League", "match_id": "77", "estado_partido": "STATUS_FINAL"})
            matches = [{"id": 77, "status": "TIMED", "utcDate": "2026-10-15T18:00:00Z"}]
            fixed = apply_reconciliation_statuses(matches, path, "Champions")
            self.assertEqual(fixed[0]["status"], "FINISHED")
            self.assertFalse(is_upcoming_match(fixed[0]))

    def test_reconciliation_promotes_stale_calendar_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reconciliacion_pendientes.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["deporte", "match_id", "estado_partido"])
                writer.writeheader()
                writer.writerow({"deporte": "LaLiga", "match_id": "42", "estado_partido": "FINISHED"})
            matches = [{"id": 42, "status": "SCHEDULED", "utcDate": "2026-10-10T18:00:00Z"}]
            fixed = apply_reconciliation_statuses(matches, path, "LaLiga")
            self.assertEqual(fixed[0]["status"], "FINISHED")
            self.assertFalse(is_upcoming_match(fixed[0]))


if __name__ == "__main__":
    unittest.main()
