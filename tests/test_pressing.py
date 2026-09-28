"""Known-answer tests for pressing metrics. All data here is synthetic, by design."""

import pandas as pd
import pytest

from regista.analytics.pressing import PRESSURE_RADIUS_M, press_windows, pressure_frames


def _frames(defender_offsets: list[list[float]], carrier_x: float = 0.0, fps: float = 1.0):
    """Home carrier h1 at (carrier_x, 0); away defenders at the given x offsets from it."""
    rows = []
    for f, offsets in enumerate(defender_offsets):
        rows.append({"frame": f, "team": "home", "player_id": "h1", "x": carrier_x, "y": 0.0})
        rows.append({"frame": f, "team": "home", "player_id": "h2", "x": carrier_x, "y": 3.0})
        for i, dx in enumerate(offsets):
            rows.append({"frame": f, "team": "away", "player_id": f"a{i}", "x": carrier_x + dx,
                         "y": 0.0})  # fmt: skip
        rows.append({"frame": f, "team": "ball", "player_id": "ball", "x": carrier_x, "y": 0.0})
    df = pd.DataFrame(rows)
    df["match_id"], df["period"] = "synthetic", 1
    df["t"] = df["frame"] / fps
    owner = (
        df[["match_id", "period", "frame"]]
        .drop_duplicates()
        .assign(owner_id="h1", owner_team="home")
    )
    return df, owner


def test_nearest_defender_and_count_within_five_yards():
    frames, owner = _frames([[1.5, 4.0, 10.0], [3.0, 30.0], [PRESSURE_RADIUS_M + 0.01]])
    p = pressure_frames(frames, owner)
    assert p["nearest_defender_m"].tolist() == pytest.approx([1.5, 3.0, PRESSURE_RADIUS_M + 0.01])
    assert p["defenders_within"].tolist() == [2, 1, 0]
    assert set(p["pressing_team"]) == {"away"}
    assert (p["carrier_id"] == "h1").all()


def test_thirds_are_named_from_the_pressing_team_view():
    # away presses and attacks -x: a home carrier near home's own goal is away's attacking third
    for carrier_x, third in ((-40.0, "attacking"), (0.0, "middle"), (40.0, "defensive")):
        frames, owner = _frames([[2.0]], carrier_x=carrier_x)
        assert pressure_frames(frames, owner)["third"].iat[0] == third


def test_press_windows_intensities():
    # 10 frames at 1 fps: 4 tight (1.5 m), 3 pressured (3 m), 3 free (10 m)
    offsets = [[1.5]] * 4 + [[3.0]] * 3 + [[10.0]] * 3
    frames, owner = _frames(offsets)
    w = press_windows(pressure_frames(frames, owner), window_s=10.0)
    row = w[(w["third"] == "all") & w["complete"]].iloc[0]
    assert row["n_frames"] == 10
    assert row["press_intensity"] == pytest.approx(0.7)
    assert row["tight_intensity"] == pytest.approx(0.4)


def test_trailing_windows_cover_only_the_past():
    offsets = [[10.0]] * 6 + [[1.0]] * 6  # pressing starts at t = 6 s
    frames, owner = _frames(offsets)
    w = press_windows(pressure_frames(frames, owner), window_s=4.0, step_s=2.0)
    w = w[w["third"] == "all"].set_index("t_end")
    assert not w.loc[2.0, "complete"]
    assert w.loc[6.0, "press_intensity"] == 0.0  # covers t in [2, 6)
    assert w.loc[8.0, "press_intensity"] == pytest.approx(0.5)  # [4, 8)
    assert w.loc[10.0, "press_intensity"] == 1.0
