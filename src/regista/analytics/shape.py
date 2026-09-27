"""Team shape per frame: defensive line height, length, width, compactness, centroid.

All metrics use each team's outfield players in its attacking frame (+x towards
the opponent goal), so home and away are directly comparable.

- ``line_height``: distance of the deepest outfield player from the team's own
  goal line (metres; own goal line is x = -52.5 in the attacking frame).
- ``length``: deepest to most advanced outfield player along x.
- ``width``: widest spread of outfield players along y.
- ``hull_area``: convex hull area of the outfield players (m^2); lower is more compact.
- ``centroid_x``/``centroid_y``: mean outfield position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from regista.analytics.formations import to_attacking_frame
from regista.schema import PITCH_LENGTH_M, Team

MIN_GK_PRESENCE = 0.5  # share of a period's frames a goalkeeper candidate must appear in
MIN_OUTFIELD = 3  # fewer visible outfield players than this: no shape for the frame

SHAPE_METRICS = ["line_height", "length", "width", "hull_area", "centroid_x", "centroid_y"]


def goalkeepers(frames: pd.DataFrame) -> pd.DataFrame:
    """Per match, period and team, the deepest regular player in the attacking frame."""
    players = to_attacking_frame(frames[frames["team"] != Team.BALL.value])
    keys = ["match_id", "period", "team"]
    n_frames = players.groupby(keys)["frame"].nunique().rename("n_frames")
    stats = players.groupby([*keys, "player_id"]).agg(x=("x", "mean"), seen=("frame", "nunique"))
    stats = stats.join(n_frames, on=keys)
    regular = stats[stats["seen"] / stats["n_frames"] >= MIN_GK_PRESENCE].reset_index()
    gk = regular.loc[regular.groupby(keys)["x"].idxmin(), [*keys, "player_id"]]
    return gk.rename(columns={"player_id": "gk_id"}).reset_index(drop=True)


def _hull_area(xy: np.ndarray) -> float:
    try:
        return float(ConvexHull(xy).volume)  # in 2-D, "volume" is the area
    except QhullError:  # collinear or duplicate points
        return 0.0


def team_shape(frames: pd.DataFrame) -> pd.DataFrame:
    """Shape metrics per match, period, frame, and team (outfield players only).

    Frames with fewer than ``MIN_OUTFIELD`` visible outfield players for a team
    are omitted for that team.
    """
    keys = ["match_id", "period", "frame", "team"]
    players = to_attacking_frame(frames[frames["team"] != Team.BALL.value])
    gk = goalkeepers(frames)
    players = players.merge(gk, on=["match_id", "period", "team"], how="left")
    outfield = players[players["player_id"] != players["gk_id"]]

    g = outfield.groupby(keys, sort=True)
    shape = g.agg(
        n=("x", "size"),
        x_min=("x", "min"),
        x_max=("x", "max"),
        y_min=("y", "min"),
        y_max=("y", "max"),
        centroid_x=("x", "mean"),
        centroid_y=("y", "mean"),
    )
    shape = shape[shape["n"] >= MIN_OUTFIELD]

    outfield = outfield.sort_values(keys)
    xy = outfield[["x", "y"]].to_numpy()
    sizes = outfield.groupby(keys, sort=True).size()
    bounds = np.r_[0, np.cumsum(sizes.to_numpy())]
    areas = pd.Series(
        [_hull_area(xy[lo:hi]) if hi - lo >= MIN_OUTFIELD else np.nan
         for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)],
        index=sizes.index,
    )  # fmt: skip

    frame_keys = ["match_id", "period", "frame"]
    times = frames.drop_duplicates(frame_keys)[[*frame_keys, "t"]]
    shape = shape.assign(
        line_height=shape["x_min"] + PITCH_LENGTH_M / 2,
        length=shape["x_max"] - shape["x_min"],
        width=shape["y_max"] - shape["y_min"],
        hull_area=areas.reindex(shape.index),
    ).reset_index()
    shape = shape.merge(times, on=frame_keys, how="left")
    return shape[[*keys, "t", "n", *SHAPE_METRICS]].rename(columns={"n": "n_outfield"})


def shape_windows(
    shape: pd.DataFrame, possession: pd.DataFrame, window_s: float = 300.0
) -> pd.DataFrame:
    """Median shape metrics per team, period, phase (in/out of possession), and window.

    ``possession`` is a per-frame team phase (``possession.team_in_possession``);
    frames with no phase are left out.
    """
    keys = ["match_id", "period", "frame"]
    s = shape.merge(possession.rename(columns={"team": "possession_team"}), on=keys)
    s = s.dropna(subset=["possession_team"])
    s["phase"] = np.where(s["possession_team"] == s["team"], "in", "out")
    s["window"] = (s["t"] // window_s).astype(int)
    group = ["match_id", "period", "team", "phase", "window"]
    out = s.groupby(group)[SHAPE_METRICS].median()
    out["n_frames"] = s.groupby(group).size()
    out = out.reset_index()
    out["t_start"] = out["window"] * window_s
    return out
