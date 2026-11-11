import unittest
from datetime import datetime, timezone, timedelta

from mlb_engine import partido_bloqueado


class TestMLBForecastLock(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 11, 18, 0, tzinfo=timezone.utc)

    def test_locks_exactly_at_30_minutes(self):
        game = {"date": (self.now + timedelta(minutes=30)).isoformat()}
        self.assertTrue(partido_bloqueado(game, self.now))

    def test_remains_editable_more_than_30_minutes_before(self):
        game = {"date": (self.now + timedelta(minutes=31)).isoformat()}
        self.assertFalse(partido_bloqueado(game, self.now))

    def test_locks_after_kickoff(self):
        game = {"date": (self.now - timedelta(minutes=1)).isoformat()}
        self.assertTrue(partido_bloqueado(game, self.now))

    def test_malformed_time_fails_closed(self):
        self.assertTrue(partido_bloqueado({"date": "unknown"}, self.now))


if __name__ == "__main__":
    unittest.main()
