"""Velocities from positions: Savitzky-Golay smoothing per object, implausible speeds masked."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from regista.schema import Team

MAX_PLAYER_SPEED_MS = 12.0  # faster than any recorded sprint; above this is tracking error


def frame_interval(frames: pd.DataFrame) -> float:
    """Seconds between consecutive frames, from the data itself."""
    g = frames.drop_duplicates(["match_id", "period", "frame"]).sort_values(
        ["match_id", "period", "frame"]
    )
    same = (g["match_id"].shift() == g["match_id"]) & (g["period"].shift() == g["period"])
    dframe = g["frame"].diff()[same]
    dt = g["t"].diff()[same]
    step = dframe > 0
    if not step.any():
        raise ValueError("need at least two frames in one period to infer the frame interval")
    return float(np.median(dt[step] / dframe[step]))


def _segment_ids(frames: pd.DataFrame) -> np.ndarray:
    """Label runs of consecutive frames for one object; a gap starts a new segment."""
    same_obj = (
        (frames["match_id"].to_numpy()[1:] == frames["match_id"].to_numpy()[:-1])
        & (frames["period"].to_numpy()[1:] == frames["period"].to_numpy()[:-1])
        & (frames["player_id"].to_numpy()[1:] == frames["player_id"].to_numpy()[:-1])
    )
    contiguous = np.diff(frames["frame"].to_numpy()) == 1
    starts = np.concatenate([[True], ~(same_obj & contiguous)])
    return np.cumsum(starts)


def add_velocities(
    frames: pd.DataFrame,
    window_s: float = 0.5,
    polyorder: int = 2,
    max_player_speed: float = MAX_PLAYER_SPEED_MS,
) -> pd.DataFrame:
    """Return frames with ``vx``/``vy`` (m/s) from a Savitzky-Golay derivative.

    Smoothing runs within each object's contiguous frame segments, never across
    gaps. Player speeds above ``max_player_speed`` are set to NaN, not clipped.
    The ball is not capped. Segments too short to fit the polynomial get NaN.
    """
    dt = frame_interval(frames)
    window = max(int(round(window_s / dt)) | 1, polyorder + 2 | 1)
    out = frames.sort_values(["match_id", "period", "player_id", "frame"]).reset_index(drop=True)
    seg = _segment_ids(out)
    x, y = out["x"].to_numpy(), out["y"].to_numpy()
    vx, vy = np.full(len(out), np.nan), np.full(len(out), np.nan)

    bounds = np.flatnonzero(np.diff(seg)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(out)], strict=True):
        n = hi - lo
        w = min(window, n if n % 2 else n - 1)
        if w <= polyorder:
            continue
        vx[lo:hi] = savgol_filter(x[lo:hi], w, polyorder, deriv=1, delta=dt)
        vy[lo:hi] = savgol_filter(y[lo:hi], w, polyorder, deriv=1, delta=dt)

    too_fast = (np.hypot(vx, vy) > max_player_speed) & (out["team"] != Team.BALL.value).to_numpy()
    vx[too_fast] = np.nan
    vy[too_fast] = np.nan
    out["vx"], out["vy"] = vx, vy
    return out


def speed(frames: pd.DataFrame) -> pd.Series:
    """Speed in m/s from ``vx``/``vy``."""
    return np.hypot(frames["vx"], frames["vy"])
