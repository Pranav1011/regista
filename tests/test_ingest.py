"""Known-answer tests for ingest conversion steps. All data here is synthetic, by design."""

import numpy as np
import pandas as pd
import pytest

from regista import io
from regista.ingest._common import (
    drop_out_of_bounds,
    home_attack_flips,
    normalise_direction,
    to_metres,
    wide_to_long,
)
from regista.schema import Source, Team, validate_frames


def test_to_metres_maps_corners_and_centre():
    # normalised: origin top-left, y down. canonical: centre origin, +y up.
    x, y = to_metres(np.array([0.0, 1.0, 0.5, 0.0]), np.array([0.0, 1.0, 0.5, 1.0]))
    np.testing.assert_allclose(x, [-52.5, 52.5, 0.0, -52.5])
    np.testing.assert_allclose(y, [34.0, -34.0, 0.0, -34.0])


def test_to_metres_rescales_any_pitch_onto_canonical():
    x, _ = to_metres(np.array([1.0]), np.array([0.5]), length=105.0)
    assert x[0] == pytest.approx(52.5)


def _wide(n_frames: int = 3, period: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_id": [period] * n_frames,
            "frame_id": np.arange(n_frames),
            "timestamp": pd.to_timedelta(np.arange(n_frames) * 0.04, unit="s"),
            "ball_x": [0.5] * n_frames,
            "ball_y": [0.5] * n_frames,
            "h1_x": [0.25, np.nan, 0.25],  # h1 not visible in frame 1
            "h1_y": [0.5, np.nan, 0.5],
            "a1_x": [0.75] * n_frames,
            "a1_y": [0.25] * n_frames,
        }
    )


def test_wide_to_long_drops_missing_objects_and_is_valid():
    frames = wide_to_long(_wide(), {"h1": Team.HOME, "a1": Team.AWAY}, "synthetic", Source.METRICA)
    validate_frames(frames)
    assert len(frames) == 3 * 3 - 1
    assert not ((frames["player_id"] == "h1") & (frames["frame"] == 1)).any()
    a1 = frames[frames["player_id"] == "a1"].iloc[0]
    assert (a1["x"], a1["y"]) == pytest.approx((26.25, 17.0))
    ball = frames[frames["team"] == "ball"]
    assert set(ball["player_id"]) == {"ball"}
    assert frames["t"].max() == pytest.approx(0.08)


def _period_frames(home_gk_x: float, period: int, away_gk_x: float | None = None) -> pd.DataFrame:
    """Two frames of a period: each team's GK at depth, outfielders near halfway."""
    away_gk_x = -home_gk_x if away_gk_x is None else away_gk_x
    rows = []
    for f in range(2):
        rows += [
            {"player_id": "h_gk", "team": "home", "x": home_gk_x, "y": 0.0},
            {"player_id": "h1", "team": "home", "x": 3.0, "y": 10.0},  # across halfway
            {"player_id": "h2", "team": "home", "x": -np.sign(home_gk_x) * 5, "y": -8.0},
            {"player_id": "a_gk", "team": "away", "x": away_gk_x, "y": 0.0},
            {"player_id": "a1", "team": "away", "x": 1.0, "y": 0.0},
        ]
        for r in rows[-5:]:
            r["frame"] = f
    df = pd.DataFrame(rows)
    df["period"] = period
    df["t"] = df["frame"] * 0.04
    return df


def test_direction_inferred_from_goalkeepers_and_normalised():
    # period 1: home GK defends -x (home attacks +x); period 2: sides swapped.
    frames = pd.concat([_period_frames(-45.0, 1), _period_frames(45.0, 2)], ignore_index=True)
    frames["vx"] = 1.0
    frames["vy"] = -2.0
    flips = home_attack_flips(frames)
    assert flips == {1: False, 2: True}

    out = normalise_direction(frames, flips)
    p2_in, p2_out = frames[frames["period"] == 2], out[out["period"] == 2]
    np.testing.assert_allclose(p2_out[["x", "y"]], -p2_in[["x", "y"]])
    np.testing.assert_allclose(p2_out[["vx", "vy"]], [[-1.0, 2.0]] * len(p2_out))
    pd.testing.assert_frame_equal(out[out["period"] == 1], frames[frames["period"] == 1])
    gk = out[out["player_id"] == "h_gk"]
    assert (gk.groupby("period")["x"].mean() < 0).all()


def test_ambiguous_direction_raises():
    same_side = _period_frames(-45.0, 1, away_gk_x=-40.0)
    with pytest.raises(ValueError, match="cannot infer attacking direction"):
        home_attack_flips(same_side)
    too_shallow = _period_frames(-10.0, 1)
    with pytest.raises(ValueError, match="cannot infer attacking direction"):
        home_attack_flips(too_shallow)


def test_drop_out_of_bounds_counts_ball_and_player_rows():
    frames = pd.DataFrame(
        {
            "team": ["home", "ball", "away", "ball", "home"],
            "x": [0.0, 58.0, -10.0, 0.0, -60.0],
            "y": [0.0, 0.0, -40.0, 38.9, 0.0],
        }
    )
    kept, dropped = drop_out_of_bounds(frames)
    assert dropped == {"ball": 1, "player": 2}
    assert kept["y"].tolist() == [0.0, 38.9]


def test_frames_roundtrip_through_io(tmp_path, monkeypatch):
    monkeypatch.setenv("REGISTA_DATA_DIR", str(tmp_path))
    frames = wide_to_long(_wide(), {"h1": Team.HOME, "a1": Team.AWAY}, "syn", Source.METRICA)
    path = io.write_frames(frames, Source.METRICA, "syn")
    assert path == tmp_path / "processed" / "metrica" / "syn.parquet"
    pd.testing.assert_frame_equal(io.read_frames(Source.METRICA, "syn"), validate_frames(frames))


def test_read_missing_frames_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("REGISTA_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="regista ingest"):
        io.read_frames(Source.METRICA, "nope")
