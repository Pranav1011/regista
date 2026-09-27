"""Known-answer tests for formation detection. All data here is synthetic, by design."""

import numpy as np
import pandas as pd
import pytest

from regista.analytics.formations import (
    TEMPLATES,
    change_points,
    classify_shape,
    detect_formations,
    window_shapes,
)


def _template_xy(name: str, rng, depth_m=35.0, width_m=45.0, noise_m=1.5) -> np.ndarray:
    """A template stretched to a plausible real size, with positional noise."""
    slots = TEMPLATES[name]
    xy = np.array([[s.x, s.y] for s in slots], dtype=float)
    xy[:, 0] *= depth_m / 40.0 * rng.uniform(0.85, 1.15)
    xy[:, 1] *= width_m / 56.0 * rng.uniform(0.85, 1.15)
    return xy + rng.normal(0, noise_m, xy.shape)


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_noisy_templates_are_recovered(name):
    rng = np.random.default_rng(42)
    for _ in range(20):
        xy = _template_xy(name, rng)
        perm = rng.permutation(len(xy))  # player order must not matter
        match = classify_shape(xy[perm])
        assert match.label == name
        roles = [s.role for s in match.slots]
        assert roles == [TEMPLATES[name][i].role for i in perm]
        assert 0.0 < match.confidence <= 1.0


def test_wrong_number_of_players_rejected():
    with pytest.raises(ValueError, match="expected 10"):
        classify_shape(np.zeros((9, 2)))


def _match_frames(home_in: str, home_out: str, n_frames=600, fps=2.0, seed=0) -> tuple:
    """Home plays home_in with the ball (first half of frames), home_out without it.

    Positions are in canonical coordinates: home attacks +x. The away team is a
    4-4-2 mirrored into its own attacking direction (-x). Each team has a GK.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        in_poss = f < n_frames // 2
        home_xy = _template_xy(home_in if in_poss else home_out, rng, noise_m=0.5) + [-25, 0]
        away_xy = -(_template_xy("4-4-2", rng, noise_m=0.5) + [-25, 0])
        for team, xy, gk_x in (("home", home_xy, -50.0), ("away", away_xy, 50.0)):
            for i, (x, y) in enumerate(xy):
                rows.append({"frame": f, "team": team, "player_id": f"{team}_{i}", "x": x, "y": y})
            rows.append({"frame": f, "team": team, "player_id": f"{team}_gk", "x": gk_x, "y": 0.0})
        rows.append({"frame": f, "team": "ball", "player_id": "ball", "x": 0.0, "y": 0.0})
    frames = pd.DataFrame(rows)
    frames["match_id"], frames["period"] = "synthetic", 1
    frames["t"] = frames["frame"] / fps
    possession = frames[["match_id", "period", "frame"]].drop_duplicates()
    possession["team"] = np.where(possession["frame"] < n_frames // 2, "home", "away")
    return frames, possession


def test_in_and_out_of_possession_detected_separately():
    frames, possession = _match_frames("4-3-3", "4-4-2")
    shapes = window_shapes(frames, possession, window_s=300, step_s=300)
    formations, roles = detect_formations(shapes)
    by = formations.set_index(["team", "phase"])["label"]
    assert by.loc[("home", "in")] == "4-3-3"
    assert by.loc[("home", "out")] == "4-4-2"
    assert by.loc[("away", "in")] == "4-4-2"  # mirrored team is read in its own direction
    assert not roles["player_id"].str.endswith("_gk").any()
    assert set(shapes["gk_id"]) == {"home_gk", "away_gk"}


def test_change_point_needs_two_consecutive_windows():
    labels = ["4-4-2", "4-4-2", "4-3-3", "4-4-2", "3-5-2", "3-5-2", "3-5-2"]
    formations = pd.DataFrame(
        {
            "match_id": "synthetic",
            "team": "home",
            "phase": "out",
            "period": 1,
            "window": range(len(labels)),
            "t_start": [300.0 * i for i in range(len(labels))],
            "label": labels,
        }
    )
    cps = change_points(formations)
    assert cps[["from_label", "to_label", "t_start"]].values.tolist() == [
        ["4-4-2", "3-5-2", 1200.0]
    ]
