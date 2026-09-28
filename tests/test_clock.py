"""Tests for match clock conversion."""

import pytest

from regista.clock import match_clock, match_seconds, parse_clock


def test_second_half_starts_at_45():
    assert match_clock(1, 0) == "00:00"
    assert match_clock(1, 125.9) == "02:05"
    assert match_clock(2, 0) == "45:00"
    assert match_clock(2, 17 * 60 + 3) == "62:03"
    assert match_seconds(2, 10) == 2710


def test_parse_clock_round_trip_and_errors():
    assert parse_clock("62:03") == 3723
    assert parse_clock("60") == 3600
    for bad in ("6x:00", "10:75", "1:2:3", ""):
        with pytest.raises(ValueError):
            parse_clock(bad)
    with pytest.raises(ValueError, match="period"):
        match_seconds(3, 0)
