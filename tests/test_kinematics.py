"""Known-answer tests for kinematics. All data here is synthetic, by design."""

import numpy as np
import pandas as pd
import pytest

from regista.analytics.kinematics import add_velocities, frame_interval, speed
from regista.schema import validate_frames

FPS = 25


def _track(player_id: str, team: str, xs: np.ndarray, ys: np.ndarray, frames=None) -> pd.DataFrame:
    frames = np.arange(len(xs)) if frames is None else np.asarray(frames)
    return pd.DataFrame(
        {
            "match_id": "synthetic",
            "period": 1,
            "frame": frames,
            "t": frames / FPS,
            "team": team,
            "player_id": player_id,
            "x": xs,
            "y": ys,
            "vx": np.nan,
            "vy": np.nan,
            "confidence": 1.0,
            "source": "metrica",
        }
    )


def test_frame_interval_from_data():
    df = _track("h1", "home", np.zeros(10), np.zeros(10))
    assert frame_interval(df) == pytest.approx(1 / FPS)


def test_constant_velocity_recovered():
    n = 100
    t = np.arange(n) / FPS
    df = _track("h1", "home", -20 + 3.0 * t, 5 - 4.0 * t)
    rng = np.random.default_rng(0)
    df["x"] += rng.normal(0, 0.02, n)  # small synthetic tracking noise
    out = validate_frames(add_velocities(df))
    assert out["vx"].median() == pytest.approx(3.0, abs=0.05)
    assert out["vy"].median() == pytest.approx(-4.0, abs=0.05)
    assert speed(out).median() == pytest.approx(5.0, abs=0.05)


def test_teleport_masked_not_clipped():
    xs = np.zeros(60)
    xs[30:] = 10.0  # a 10 m jump in one frame: tracking error
    out = add_velocities(_track("h1", "home", xs, np.zeros(60)))
    near_jump = out["frame"].between(27, 33)
    assert out.loc[near_jump, "vx"].isna().any()
    assert out.loc[~out["frame"].between(20, 40), "vx"].abs().max() < 1e-9
    assert not (speed(out) > 12.0).any()


def test_ball_is_not_capped():
    t = np.arange(50) / FPS
    out = add_velocities(_track("ball", "ball", -20 + 25.0 * t, np.zeros(50)))
    assert out["vx"].median() == pytest.approx(25.0, abs=0.1)


def test_smoothing_never_crosses_a_gap():
    # frames 0-19 stationary at x=0, then frames 40-59 stationary at x=30
    frames = np.r_[np.arange(20), np.arange(40, 60)]
    xs = np.r_[np.zeros(20), np.full(20, 30.0)]
    out = add_velocities(_track("h1", "home", xs, np.zeros(40), frames=frames))
    assert out["vx"].abs().max() < 1e-9


def test_short_segment_gets_nan():
    out = add_velocities(
        pd.concat(
            [
                _track("h1", "home", np.zeros(30), np.zeros(30)),
                _track("h2", "home", np.zeros(2), np.zeros(2)),
            ]
        )
    )
    assert out.loc[out["player_id"] == "h2", "vx"].isna().all()
    assert out.loc[out["player_id"] == "h1", "vx"].notna().all()
