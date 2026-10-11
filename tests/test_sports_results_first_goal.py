from sports_results import _first_scoring_team_from_goals


def test_first_goal_uses_earliest_official_goal():
    goals = [
        {"minute": 87, "injuryTime": 0, "team": {"name": "Atletico Madrid"}},
        {"minute": 33, "injuryTime": 0, "team": {"name": "Alaves"}},
        {"minute": 45, "injuryTime": 6, "team": {"name": "Alaves"}},
    ]
    assert _first_scoring_team_from_goals(goals) == "Alaves"


def test_first_goal_handles_injury_time_after_regular_minute():
    goals = [
        {"minute": 45, "injuryTime": 6, "team": {"name": "Alaves"}},
        {"minute": 45, "injuryTime": 1, "team": {"name": "Atletico Madrid"}},
    ]
    assert _first_scoring_team_from_goals(goals) == "Atletico Madrid"


def test_first_goal_refuses_ambiguous_same_timestamp():
    goals = [
        {"minute": 33, "injuryTime": 0, "team": {"name": "Alaves"}},
        {"minute": 33, "injuryTime": 0, "team": {"name": "Atletico Madrid"}},
    ]
    assert _first_scoring_team_from_goals(goals) is None


def test_first_goal_returns_none_when_feed_has_no_team():
    assert _first_scoring_team_from_goals([{"minute": 33, "team": {}}]) is None
    assert _first_scoring_team_from_goals(None) is None
