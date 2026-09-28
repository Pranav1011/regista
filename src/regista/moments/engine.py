"""Causal streaming engine: per-minute trailing-window features computed from past data only.

The engine walks each period in chunks of ``step_s`` seconds. At the end of a
chunk (time ``t_end``) it has seen only rows with ``t < t_end``: kinematics are
recomputed per chunk with a short past context (never future frames), the ball
owner uses a radius estimated from kick onsets seen so far, and window features
cover ``[t_end - window_s, t_end)``. Output up to any time is therefore
identical whether or not later data exists.

Window features per team: out-of-possession and in-possession formation label,
runner-up, margin and back-line count; press intensity (as the pressing team);
median defensive line height out of possession.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regista.analytics.formations import back_line, detect_formations, window_shapes
from regista.analytics.kinematics import add_velocities
from regista.analytics.possession import ball_owner, kick_onset_distances, team_in_possession
from regista.analytics.pressing import PRESSURE_RADIUS_M, TIGHT_PRESSURE_M, pressure_frames
from regista.analytics.shape import team_shape
from regista.pipeline import scaled_frames

KEYS = ["match_id", "period", "frame"]
TEAMS = ("home", "away")


@dataclass(frozen=True)
class StreamConfig:
    window_s: float = 300.0
    step_s: float = 60.0
    context_s: float = 1.0  # past-only context for velocity smoothing
    min_onsets: int = 30  # kick onsets needed before the radius is estimated from data
    radius_prior_m: float = 0.5  # the frozen v1 radius, used until then


def _one_window(rows: pd.DataFrame, phase: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Formations for exactly one window covering all given rows."""
    return detect_formations(window_shapes(rows, phase, window_s=1e9, step_s=1e9))


def stream_windows(frames: pd.DataFrame, params: dict, config: StreamConfig | None = None
                   ) -> pd.DataFrame:  # fmt: skip
    """Per-minute trailing-window features for one match, computed causally.

    ``frames`` are canonical frames (velocities are recomputed per chunk).
    Returns one row per period, window end, and team.
    """
    cfg = config or StreamConfig()
    pp = params["v2"]
    match_id = frames["match_id"].iat[0]
    onsets: list[float] = []
    rows = []
    for period, pf in frames.groupby("period", sort=True):
        pf = pf.sort_values(["t", "player_id"])
        buf_frames, buf_phase, buf_pressure, buf_shape = [], [], [], []
        owner_tail = None
        t_last = float(pf["t"].max())
        n_steps = int(np.ceil((t_last + 1e-9) / cfg.step_s))
        for k in range(1, n_steps + 1):
            t_end = k * cfg.step_s
            t_chunk = t_end - cfg.step_s
            seen = pf[(pf["t"] >= t_chunk - cfg.context_s) & (pf["t"] < t_end)]
            if seen["frame"].nunique() < 3:
                continue
            chunk = add_velocities(seen)
            chunk = chunk[chunk["t"] >= t_chunk]
            if chunk["frame"].nunique() < 3:  # e.g. the last few frames of a period
                continue
            onsets.extend(kick_onset_distances(chunk).dropna().tolist())
            radius = (float(np.quantile(onsets, pp["radius_quantile"]))
                      if len(onsets) >= cfg.min_onsets else cfg.radius_prior_m)  # fmt: skip
            owner = ball_owner(chunk, radius, pp["max_ball_speed"])
            gap = scaled_frames(pp["max_gap_frames"], chunk)
            with_tail = owner if owner_tail is None else pd.concat([owner_tail, owner])
            phase = team_in_possession(with_tail, gap)
            phase = phase[phase["frame"].isin(owner["frame"])]
            owner_tail = owner[owner["frame"] >= owner["frame"].max() - (gap or 0)]

            buf_frames.append(chunk)
            buf_phase.append(phase)
            buf_pressure.append(pressure_frames(chunk, owner))
            possession = phase.rename(columns={"team": "possession_team"})
            buf_shape.append(team_shape(chunk).merge(possession, on=KEYS, how="left"))
            keep = int(np.ceil(cfg.window_s / cfg.step_s))
            buf_frames, buf_phase = buf_frames[-keep:], buf_phase[-keep:]
            buf_pressure, buf_shape = buf_pressure[-keep:], buf_shape[-keep:]

            t_start = max(t_end - cfg.window_s, 0.0)
            wf = pd.concat(buf_frames)
            wf = wf[wf["t"] >= t_start]
            wp = pd.concat(buf_phase)
            wp = wp[wp["frame"].isin(wf["frame"])]
            forms, _ = _one_window(wf, wp)
            press = pd.concat(buf_pressure)
            press = press[press["t"] >= t_start]
            shape = pd.concat(buf_shape)
            shape = shape[shape["t"] >= t_start]
            last_frame = int(wf["frame"].max())
            for team in TEAMS:
                row = {"match_id": match_id, "period": int(period), "t_start": t_start,
                       "t_end": t_end, "emit_frame": last_frame,
                       "complete": t_end - cfg.window_s >= 0, "team": team,
                       "radius_m": radius}  # fmt: skip
                for phase_name in ("out", "in"):
                    f = forms[(forms["team"] == team) & (forms["phase"] == phase_name)]
                    label = f["label"].iat[0] if len(f) and pd.notna(f["label"].iat[0]) else None
                    row[f"label_{phase_name}"] = label
                    row[f"runner_up_{phase_name}"] = f["runner_up"].iat[0] if label else None
                    row[f"margin_{phase_name}"] = float(f["margin"].iat[0]) if label else np.nan
                    row[f"back_line_{phase_name}"] = back_line(label) if label else np.nan
                p = press[press["pressing_team"] == team]
                row["press_frames"] = len(p)
                row["press_intensity"] = (
                    float((p["nearest_defender_m"] <= PRESSURE_RADIUS_M).mean())
                    if len(p)
                    else np.nan
                )
                row["tight_intensity"] = (
                    float((p["nearest_defender_m"] <= TIGHT_PRESSURE_M).mean())
                    if len(p)
                    else np.nan
                )
                s = shape[(shape["team"] == team) & shape["possession_team"].notna()
                          & (shape["possession_team"] != team)]  # fmt: skip
                row["line_height_out"] = float(s["line_height"].median()) if len(s) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)
