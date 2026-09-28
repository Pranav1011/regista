"""Known-answer and causality tests for moment detectors. All data here is synthetic, by design."""

import json
from pathlib import Path

import pandas as pd
import pytest
from synthetic_match import MatchSpec, make_match

from regista.moments import DetectorConfig, detect_moments, stream_windows

PARAMS = json.loads(
    (Path(__file__).resolve().parent.parent / "eval" / "params_phase1.json").read_text()
)
CFG = DetectorConfig(persistence_min=3, min_margin=0.03, press_delta=0.12, line_delta_m=5.0)
CHANGE_T = 600.0  # injected changes happen 10 minutes into period 1
FAST = {"period_s": 1200.0, "fps": 4.0}


def _moments(spec: MatchSpec) -> pd.DataFrame:
    return detect_moments(stream_windows(make_match(spec), PARAMS), CFG)


def _assert_single(m: pd.DataFrame, kind: str, team: str) -> pd.Series:
    hits = m[m["type"] == kind]
    assert len(hits) == 1, f"expected one {kind}, got:\n{m[['type', 'team', 'period', 'emit_t']]}"
    hit = hits.iloc[0]
    assert hit["team"] == team and hit["period"] == 1
    # the start estimate is within half a window of the true change,
    assert abs(hit["start_t"] - CHANGE_T) <= CFG.window_s / 2
    # and it fires after the change, within a window plus the persistence plus one step
    limit = CHANGE_T + CFG.window_s + (CFG.persistence_min + 1) * CFG.step_s
    assert CHANGE_T < hit["emit_t"] <= limit
    return hit


@pytest.fixture(scope="module")
def clean_moments():
    return _moments(MatchSpec(**FAST))


def test_clean_match_fires_nothing(clean_moments):
    assert clean_moments.empty, clean_moments[["type", "team", "period", "emit_t"]]


def test_back_four_to_back_five_fires_once():
    five = lambda p, t: "5-3-2" if p == 2 or t >= CHANGE_T else "4-4-2"  # noqa: E731
    m = _moments(MatchSpec(**FAST, away_formation=five))
    hit = _assert_single(m, "back_line_change", "away")
    assert (hit["evidence"]["before"], hit["evidence"]["after"]) == (4, 5)


def test_press_drop_fires_once():
    press = lambda p, t: 1.5 if p == 1 and t < CHANGE_T else None  # noqa: E731
    m = _moments(MatchSpec(**FAST, away_press=press))
    hit = _assert_single(m, "press_change", "away")
    assert hit["evidence"]["after"] < hit["evidence"]["before"] - CFG.press_delta


def test_line_drop_fires_once():
    line = lambda p, t: -12.0 if p == 2 or t >= CHANGE_T else 0.0  # noqa: E731
    m = _moments(MatchSpec(**FAST, home_line=line))
    hit = _assert_single(m, "line_height_shift", "home")
    assert hit["evidence"]["after"] < hit["evidence"]["before"] - CFG.line_delta_m


@pytest.mark.parametrize("cut", [(1, 1000.0), (2, 300.0)])
def test_output_up_to_t_is_identical_without_later_data(cut):
    five = lambda p, t: "5-3-2" if p == 2 or t >= CHANGE_T else "4-4-2"  # noqa: E731
    press = lambda p, t: 1.5 if p == 1 and t < CHANGE_T else None  # noqa: E731
    frames = make_match(MatchSpec(**FAST, away_formation=five, away_press=press))
    period, t = cut
    before = (frames["period"] < period) | ((frames["period"] == period) & (frames["t"] < t))
    full_w = stream_windows(frames, PARAMS)
    cut_w = stream_windows(frames[before], PARAMS)

    def upto(w: pd.DataFrame) -> pd.DataFrame:
        keep = (w["period"] < period) | ((w["period"] == period) & (w["t_end"] <= t))
        return w[keep].reset_index(drop=True)

    pd.testing.assert_frame_equal(upto(full_w), upto(cut_w))

    def moments_upto(w: pd.DataFrame) -> pd.DataFrame:
        m = detect_moments(w, CFG)
        keep = (m["period"] < period) | ((m["period"] == period) & (m["emit_t"] <= t))
        return m[keep].reset_index(drop=True)

    full_m, cut_m = moments_upto(full_w), moments_upto(cut_w)
    assert not full_m.empty  # the check is not vacuous
    pd.testing.assert_frame_equal(full_m, cut_m)
