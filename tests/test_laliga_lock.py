import unittest
from datetime import datetime, timedelta, timezone

from laliga_engine import partido_bloqueado


class TestLaLigaForecastLock(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 11, 18, 0, tzinfo=timezone.utc)

    def test_locks_exactly_at_30_minutes(self):
        partido = {"utcDate": (self.now + timedelta(minutes=30)).isoformat()}
        self.assertTrue(partido_bloqueado(partido, self.now))

    def test_remains_editable_31_minutes_before(self):
        partido = {"utcDate": (self.now + timedelta(minutes=31)).isoformat()}
        self.assertFalse(partido_bloqueado(partido, self.now))

    def test_locks_after_kickoff(self):
        partido = {"utcDate": (self.now - timedelta(minutes=1)).isoformat()}
        self.assertTrue(partido_bloqueado(partido, self.now))

    def test_naive_datetime_is_treated_as_utc(self):
        partido = {"utcDate": (self.now + timedelta(minutes=30)).replace(tzinfo=None).isoformat()}
        self.assertTrue(partido_bloqueado(partido, self.now))

    def test_malformed_time_fails_closed(self):
        self.assertTrue(partido_bloqueado({"utcDate": "unknown"}, self.now))


if __name__ == "__main__":
    unittest.main()
