import unittest
from datetime import datetime, timezone, timedelta
from sports_context_reasoner import analyze_news


class SportsContextReasonerTests(unittest.TestCase):
    def test_recent_relevant_injury_is_flagged_but_does_not_change_probability(self):
        now = datetime(2026, 10, 12, tzinfo=timezone.utc)
        result = analyze_news([{
            "Titular": "Houston Astros pitcher injury update",
            "Fuente": "Example Sports",
            "Enlace": "https://example.com/news",
            "_published": now - timedelta(hours=3),
        }], entities=["Houston Astros"], now=now)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertEqual(result["fresh_count"], 1)
        self.assertIsNone(result["probability_adjustment"])
        self.assertEqual(result["decision"], "NO_AUTOMATIC_ADJUSTMENT")
        self.assertIn("injury", result["items"][0]["signals"])

    def test_no_news_means_unknown_not_no_injuries(self):
        result = analyze_news([], entities=["Real Madrid"])
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertIn("no puede concluirse", result["summary"])
        self.assertIsNone(result["probability_adjustment"])

    def test_old_news_is_not_treated_as_current(self):
        now = datetime(2026, 10, 12, tzinfo=timezone.utc)
        result = analyze_news([{
            "title": "Real Madrid lineup news",
            "url": "https://example.com",
            "published": now - timedelta(days=8),
        }], entities=["Real Madrid"], now=now)
        self.assertEqual(result["status"], "STALE_OR_UNDATED")
        self.assertEqual(result["fresh_count"], 0)

    def test_news_without_entity_match_needs_relevance_check(self):
        now = datetime(2026, 10, 12, tzinfo=timezone.utc)
        result = analyze_news([{
            "title": "Football injury update",
            "published": now - timedelta(hours=1),
        }], entities=["Real Madrid"], now=now)
        self.assertEqual(result["status"], "RELEVANCE_UNCONFIRMED")


if __name__ == "__main__":
    unittest.main()
