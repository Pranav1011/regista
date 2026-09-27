"""Known-answer tests for pitch control and passing options. All data is synthetic, by design."""

import numpy as np
import pandas as pd
import pytest

from regista.analytics.passing_options import (
    Scene,
    build_scene,
    default_params,
    passing_options,
    pitch_control_at,
    remove_offside,
    to_scene_coords,
)

P = default_params()


def _scene(att, dfn, ball=(0.0, 0.0), gk=None) -> Scene:
    att, dfn = np.array(att, float), np.array(dfn, float).reshape(-1, 2)
    return Scene(
        att_ids=np.array([f"a{i}" for i in range(len(att))]),
        att_pos=att,
        att_vel=np.zeros_like(att),
        def_pos=dfn,
        def_vel=np.zeros_like(dfn),
        def_is_gk=np.array(gk if gk is not None else [False] * len(dfn)),
        ball=np.array(ball, float),
    )


def test_open_teammate_ranks_above_marked_teammate():
    # carrier at the ball; a1 unmarked 15 m away; a2 15 m away with a defender at 1 m
    scene = _scene(att=[(0, 0), (15, 10), (15, -10)], dfn=[(15, -11), (-30, 0)])
    options = passing_options(scene, P).set_index("player_id")["pitch_control"]
    assert "a0" not in options.index  # the carrier is not an option
    assert options["a1"] > 0.9
    # at the receiver's feet with a defender 1 m away the model gives about an even contest
    assert options["a2"] < 0.6
    assert options["a1"] - options["a2"] > 0.3


def test_control_is_near_one_at_attacker_and_near_zero_at_defender():
    scene = _scene(att=[(0, 0), (20, 0)], dfn=[(-20, 0)])
    assert pitch_control_at(np.array([20.0, 0.0]), scene, P) > 0.95
    assert pitch_control_at(np.array([-20.0, 0.0]), scene, P) < 0.05


def test_equidistant_duel_is_even():
    scene = _scene(att=[(0, 10)], dfn=[(0, -10)], ball=(-30, 0))
    assert pitch_control_at(np.array([0.0, 0.0]), scene, P) == pytest.approx(0.5, abs=0.02)


def test_goalkeeper_wins_more_contested_balls():
    base = _scene(att=[(0, 10)], dfn=[(0, -10)], ball=(-30, 0))
    with_gk = _scene(att=[(0, 10)], dfn=[(0, -10)], ball=(-30, 0), gk=[True])
    target = np.array([0.0, 0.0])
    assert pitch_control_at(target, with_gk, P) < pitch_control_at(target, base, P)


def test_offside_attackers_removed():
    scene = _scene(att=[(0, 0), (30, 5), (10, 0)], dfn=[(20, 0), (45, 0)])
    onside = remove_offside(scene, tol=0.2)
    assert list(onside.att_ids) == ["a0", "a2"]
    assert onside.offside_ids == ["a1"]


def _frames(attacking_team: str) -> pd.DataFrame:
    rows = [
        ("home", "h1", -10.0, 5.0),
        ("home", "h_gk", -50.0, 0.0),
        ("away", "a1", 10.0, -5.0),
        ("away", "a_gk", 50.0, 0.0),
        ("ball", "ball", -10.0, 5.0 if attacking_team == "home" else -5.0),
    ]
    df = pd.DataFrame(rows, columns=["team", "player_id", "x", "y"])
    df["vx"], df["vy"] = 1.0, 0.0
    df["match_id"], df["period"], df["frame"] = "synthetic", 1, 0
    return df


def test_scene_is_in_attacking_frame_for_away_team():
    gks = {"home": "h_gk", "away": "a_gk"}
    home = build_scene(_frames("home"), 1, 0, "home", gks, offside=False)
    away = build_scene(_frames("away"), 1, 0, "away", gks, offside=False)
    np.testing.assert_allclose(home.att_pos, [[-10, 5], [-50, 0]])
    np.testing.assert_allclose(away.att_pos, [[-10, 5], [-50, 0]])  # away rotated 180 degrees
    np.testing.assert_allclose(away.att_vel, [[-1, 0], [-1, 0]])
    assert away.def_is_gk.tolist() == [False, True]
    np.testing.assert_allclose(to_scene_coords([10.0, -5.0], "away"), [-10.0, 5.0])
