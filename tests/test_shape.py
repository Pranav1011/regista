"""Known-answer tests for team shape metrics. All data here is synthetic, by design."""

import numpy as np
import pandas as pd
import pytest

from regista.analytics.shape import goalkeepers, shape_windows, team_shape


def _frames(n_frames: int = 3) -> pd.DataFrame:
    """Home outfield players on a 20 m x 30 m rectangle (corners + centre), GK deep.

    Home attacks +x. Away is the same shape mirrored into its own direction (-x).
    """
    home = [(-30.0, -15.0), (-30.0, 15.0), (-10.0, -15.0), (-10.0, 15.0), (-20.0, 0.0)]
    rows = []
    for f in range(n_frames):
        for i, (x, y) in enumerate(home):
            rows.append({"frame": f, "team": "home", "player_id": f"h{i}", "x": x, "y": y})
            rows.append({"frame": f, "team": "away", "player_id": f"a{i}", "x": -x, "y": -y})
        rows.append({"frame": f, "team": "home", "player_id": "h_gk", "x": -50.0, "y": 0.0})
        rows.append({"frame": f, "team": "away", "player_id": "a_gk", "x": 50.0, "y": 0.0})
        rows.append({"frame": f, "team": "ball", "player_id": "ball", "x": 0.0, "y": 0.0})
    df = pd.DataFrame(rows)
    df["match_id"], df["period"] = "synthetic", 1
    df["t"] = df["frame"] * 60.0
    return df


def test_goalkeepers_are_deepest_regular_players():
    gk = goalkeepers(_frames()).set_index("team")["gk_id"]
    assert gk.to_dict() == {"away": "a_gk", "home": "h_gk"}


def test_shape_metrics_known_rectangle():
    shape = team_shape(_frames())
    assert len(shape) == 3 * 2
    for team in ("home", "away"):  # mirrored away team must read identically
        s = shape[shape["team"] == team].iloc[0]
        assert s["n_outfield"] == 5
        assert s["line_height"] == pytest.approx(-30.0 + 52.5)
        assert s["length"] == pytest.approx(20.0)
        assert s["width"] == pytest.approx(30.0)
        assert s["hull_area"] == pytest.approx(600.0)
        assert (s["centroid_x"], s["centroid_y"]) == pytest.approx((-20.0, 0.0))


def test_too_few_outfield_players_gives_no_shape():
    frames = _frames(1)
    frames = frames[~frames["player_id"].isin(["h0", "h1", "h2"])]
    shape = team_shape(frames)
    assert set(shape["team"]) == {"away"}


def test_shape_windows_split_by_phase():
    frames = _frames(4)
    frames.loc[(frames["frame"] >= 2) & (frames["player_id"] == "h0"), "x"] = -40.0
    possession = frames[["match_id", "period", "frame"]].drop_duplicates()
    possession["team"] = np.where(possession["frame"] < 2, "home", "away")
    w = shape_windows(team_shape(frames), possession, window_s=600.0)
    home = w[w["team"] == "home"].set_index("phase")
    assert home.loc["in", "line_height"] == pytest.approx(22.5)
    assert home.loc["out", "line_height"] == pytest.approx(12.5)
    assert home.loc["in", "n_frames"] == 2
