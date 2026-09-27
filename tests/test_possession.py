"""Known-answer tests for possession and pass inference. All data here is synthetic, by design."""

import numpy as np
import pandas as pd
import pytest

from regista.analytics.possession import ball_owner, detect_passes, score_passes


def _scene(ball_xy, players: dict[str, tuple[str, list]], ball_v=None) -> pd.DataFrame:
    """Per-frame scene: ball_xy is a list of (x, y); players maps id -> (team, [(x, y), ...])."""
    rows = []
    for f, (bx, by) in enumerate(ball_xy):
        vx, vy = ball_v[f] if ball_v is not None else (0.0, 0.0)
        rows.append({"frame": f, "team": "ball", "player_id": "ball", "x": bx, "y": by,
                     "vx": vx, "vy": vy})  # fmt: skip
        for pid, (team, xy) in players.items():
            rows.append({"frame": f, "team": team, "player_id": pid, "x": xy[f][0],
                         "y": xy[f][1], "vx": np.nan, "vy": np.nan})  # fmt: skip
    df = pd.DataFrame(rows)
    df["match_id"], df["period"] = "synthetic", 1
    return df


def _pass_scene(receiver_team: str) -> pd.DataFrame:
    # h1 at (0,0) holds the ball for frames 0-4, ball travels frames 5-9, receiver at (10,0).
    ball = [(0.0, 0.0)] * 5 + [(2.0 * i, 0.0) for i in range(1, 5)] + [(10.0, 0.0)] * 6
    n = len(ball)
    players = {
        "h1": ("home", [(0.0, 0.3)] * n),
        "r": (receiver_team, [(10.0, 0.3)] * n),
    }
    return _scene(ball, players)


def test_owner_is_nearest_player_within_radius():
    owner = ball_owner(_pass_scene("home"), radius_m=1.0)
    assert owner.loc[owner["frame"] < 5, "owner_id"].eq("h1").all()
    assert owner.loc[owner["frame"].between(5, 8), "owner_id"].isna().all()
    assert owner.loc[owner["frame"] >= 9, "owner_id"].eq("r").all()


def test_fast_ball_has_no_owner():
    n = 3
    scene = _scene(
        [(0.0, 0.0)] * n,
        {"h1": ("home", [(0.0, 0.2)] * n)},
        ball_v=[(0.0, 0.0), (20.0, 0.0), (np.nan, np.nan)],
    )
    owner = ball_owner(scene, radius_m=1.0, max_ball_speed=10.0)
    assert owner["owner_id"].tolist()[0] == "h1"
    assert owner["owner_id"].isna().tolist() == [False, True, True]


def test_pass_between_teammates_detected():
    passes = detect_passes(ball_owner(_pass_scene("home"), radius_m=1.0))
    assert len(passes) == 1
    p = passes.iloc[0]
    assert (p["kind"], p["from_player"], p["to_player"]) == ("pass", "h1", "r")
    assert (p["start_frame"], p["end_frame"]) == (4, 9)


def test_change_to_opponent_is_turnover():
    passes = detect_passes(ball_owner(_pass_scene("away"), radius_m=1.0))
    assert passes["kind"].tolist() == ["turnover"]


def test_min_hold_filters_brief_touches_and_max_gap_breaks_chains():
    owner = pd.DataFrame(
        {
            "match_id": "synthetic",
            "period": 1,
            "frame": np.arange(12),
            "owner_id": ["a"] * 4 + ["x"] + ["a"] * 2 + [None] * 3 + ["b"] * 2,
            "owner_team": ["home"] * 4 + ["away"] + ["home"] * 2 + [None] * 3 + ["home"] * 2,
        }
    )
    passes = detect_passes(owner, min_hold_frames=2)
    assert passes[["from_player", "to_player"]].values.tolist() == [["a", "b"]]
    assert detect_passes(owner, min_hold_frames=2, max_gap_frames=3).empty


def test_passes_do_not_cross_periods():
    owner = pd.DataFrame(
        {
            "match_id": "synthetic",
            "period": [1, 1, 2, 2],
            "frame": [0, 1, 2, 3],
            "owner_id": ["a", "a", "b", "b"],
            "owner_team": ["home"] * 4,
        }
    )
    assert detect_passes(owner).empty


def test_score_passes_tolerance_and_one_to_one():
    truth = pd.DataFrame(
        {
            "period": [1, 1, 1],
            "from_player": ["a", "a", "b"],
            "to_player": ["b", "b", "c"],
            "start_frame": [100, 110, 200],
        }  # fmt: skip
    )
    pred = pd.DataFrame(
        {
            "period": [1, 1, 1, 1],
            "from_player": ["a", "a", "b", "c"],
            "to_player": ["b", "b", "c", "a"],
            "start_frame": [101, 103, 230, 300],
        }  # fmt: skip
    )
    s = score_passes(pred, truth, tol_frames=5)
    # 101 matches 100; 103 cannot also take 100 and is 7 frames from 110; 230 is too late.
    assert (s.tp, s.fp, s.fn) == (1, 3, 2)
    assert s.precision == pytest.approx(0.25)
    assert s.recall == pytest.approx(1 / 3)
    assert s.matched_true.tolist() == [True, False, False]
    s = score_passes(pred, truth, tol_frames=8)
    assert (s.tp, s.fp, s.fn) == (2, 2, 1)


def test_frames_without_ball_have_no_owner_and_owner_is_not_carried_forward():
    n = 5
    scene = _scene([(0.0, 0.0)] * n, {"h1": ("home", [(0.0, 0.2)] * n)})
    # frame 2: ball not tracked at all; frame 3: ball far from everyone
    scene = scene[~((scene["frame"] == 2) & (scene["team"] == "ball"))]
    scene.loc[(scene["frame"] == 3) & (scene["team"] == "ball"), "x"] = 20.0
    owner = ball_owner(scene, radius_m=1.0)
    assert owner["frame"].tolist() == [0, 1, 2, 3, 4]
    assert owner["ball_visible"].tolist() == [True, True, False, True, True]
    assert owner["owner_id"].isna().tolist() == [False, False, True, True, False]
