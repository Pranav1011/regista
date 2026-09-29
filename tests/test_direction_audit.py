"""Direction-word audit rules on synthetic sentences."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "agent"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from direction_audit import check_sentence  # noqa: E402


def verdicts(s):
    return [v for v, _ in check_sentence(s)]


def test_team_comparison_direction():
    assert verdicts("Home sat deeper at 30.1 m while away held 35.2 m.") == ["ok"]
    assert verdicts("Away sat deeper at 35.13m than home's 34.065m.") == ["error"]
    assert verdicts("The home team held a higher line of 43.97 m compared to the away team's "
                    "32.64 m (a difference of 11.33 m).") == ["ok"]  # fmt: skip


def test_change_direction_and_difference_is_not_a_team_value():
    assert verdicts("Their line height rose from 22.57 to 44.04 metres.") == ["ok"]
    assert verdicts("They dropped deep, from 6.01m to 45.9m.") == ["error"]
    assert verdicts("Away's line was higher by 4.13 m than home's 33.25 m.") == ["unchecked"]


def test_back_line_counts_are_ignored():
    assert verdicts("The back line dropped from 5 to 3 defenders.") == []
