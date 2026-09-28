"""Tests for match clock conversion."""

import pytest

from regista.clock import match_clock, match_seconds, parse_clock


def test_second_half_starts_at_45_and_stoppage_is_explicit():
    assert match_clock(1, 0) == "00:00"
    assert match_clock(1, 125.9) == "02:05"
    assert match_clock(1, 2820) == "45+2:00"  # first-half stoppage, not 47:00
    assert match_clock(2, 0) == "45:00"
    assert match_clock(2, 17 * 60 + 3) == "62:03"
    assert match_clock(2, 2760) == "90+1:00"
    assert match_clock(2, 2700) == "90:00"
    assert match_clock(1, 2700) == "45+0:00"
    assert match_seconds(2, 10) == 2710


def test_parse_clock():
    assert parse_clock("62:03") == (None, 3723)
    assert parse_clock("60") == (None, 3600)
    assert parse_clock("45+2:10") == (1, 2830)
    assert parse_clock("90+1") == (2, 2760)
    for bad in ("6x:00", "10:75", "1:2:3", "", "46+1:00"):
        with pytest.raises(ValueError):
            parse_clock(bad)
    with pytest.raises(ValueError, match="period"):
        match_seconds(3, 0)
