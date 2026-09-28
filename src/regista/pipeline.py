"""The frozen Phase 1 pipeline, shared by the match store and the eval scripts.

Parameters come from ``eval/params_phase1.json`` (tuned on Metrica games 1-2)
and are passed in as a dict, so this module does no file I/O. Frame-count
parameters were tuned at 25 fps and are converted by duration for other rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regista.analytics.kinematics import frame_interval
from regista.analytics.passing_options import (
    build_scene,
    default_params,
    pitch_control_at,
    to_scene_coords,
)
from regista.analytics.possession import (
    ball_owner,
    detect_passes,
    detect_passes_v2,
    match_radius,
    release_features,
    team_in_possession,
)
from regista.analytics.shape import goalkeepers
from regista.schema import Team

TUNED_FPS = 25.0  # Metrica frame rate; frame-count parameters were tuned at this rate


def scaled_frames(n_frames_at_tuned_fps: int | None, frames: pd.DataFrame) -> int | None:
    """Convert a frame count tuned at 25 fps to the same duration at this data's rate."""
    if n_frames_at_tuned_fps is None:
        return None
    return max(int(round(n_frames_at_tuned_fps / TUNED_FPS / frame_interval(frames))), 1)


def v2_owner(frames: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, float]:
    """Frame-level owner with the v2 per-match radius; returns (owner, radius)."""
    radius = match_radius(frames, params["v2"]["radius_quantile"])
    return ball_owner(frames, radius, params["v2"]["max_ball_speed"]), radius


def possession_phase(frames: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Per-frame team in possession, using the v2 owner and its max gap."""
    owner, _ = v2_owner(frames, params)
    return team_in_possession(owner, scaled_frames(params["v2"]["max_gap_frames"], frames))


def detect(
    frames: pd.DataFrame, version: str, params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """(owner table, detected passes and turnovers, radius used) for one match."""
    if version == "v1":
        pp = params["v1"]["possession"]
        radius = pp["radius_m"]
        owner = ball_owner(frames, radius, pp["max_ball_speed"])
        return owner, detect_passes(owner, pp["min_hold_frames"], pp["max_gap_frames"]), radius
    if version == "v2":
        pp = params["v2"]
        owner, radius = v2_owner(frames, params)
        dt = frame_interval(frames)
        detected = detect_passes_v2(
            frames,
            owner,
            pp["min_hold_frames"],
            pp["max_gap_frames"],
            pp["release_speed"],
            pp["release_cos"],
            back_frames=round(pp["release_back_s"] / dt),
            fwd_frames=round(pp["release_fwd_s"] / dt),
        )
        return owner, detected, radius
    raise ValueError(f"unknown pass detector version {version!r}")


def pass_attempts(frames: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Pass attempts from v2: passes, plus turnovers where the ball was released.

    A turnover counts as a failed attempt only if the ball accelerated away from
    the owner (v2's frozen release gate); its start moves to the release frame.
    Start and end points are tracked ball positions. Columns: period, frame
    (release), end_frame, team, from_player, to_player, start/end x/y, completed.
    """
    _, detected, _ = detect(frames, "v2", params)
    pp = params["v2"]
    dt = frame_interval(frames)
    turnovers = detected[detected["kind"] == "turnover"]
    feats = release_features(frames, turnovers, round(pp["release_back_s"] / dt),
                             round(pp["release_fwd_s"] / dt))  # fmt: skip
    ok = feats[(feats["speed"] >= pp["release_speed"]) & (feats["cos"] >= pp["release_cos"])]
    release = ok.groupby("pass_idx")["frame"].min()
    failed = turnovers.loc[turnovers.index.intersection(release.index)].copy()
    failed["start_frame"] = release.loc[failed.index].to_numpy()
    attempts = pd.concat([detected[detected["kind"] == "pass"], failed])

    ball = frames[frames["team"] == Team.BALL.value].set_index(["period", "frame"])[["x", "y"]]
    start = ball.reindex(pd.MultiIndex.from_arrays([attempts["period"], attempts["start_frame"]]))
    end = ball.reindex(pd.MultiIndex.from_arrays([attempts["period"], attempts["end_frame"]]))
    out = pd.DataFrame(
        {
            "period": attempts["period"].astype(int).to_numpy(),
            "frame": attempts["start_frame"].astype(int).to_numpy(),
            "end_frame": attempts["end_frame"].astype(int).to_numpy(),
            "team": attempts["team"].astype(str).to_numpy(),
            "from_player": attempts["from_player"].astype(str).to_numpy(),
            "to_player": attempts["to_player"].astype(str).to_numpy(),
            "start_x": start["x"].to_numpy(),
            "start_y": start["y"].to_numpy(),
            "end_x": end["x"].to_numpy(),
            "end_y": end["y"].to_numpy(),
            "completed": (attempts["kind"] == "pass").astype(int).to_numpy(),
        }
    )
    return out.dropna(subset=["start_x", "end_x"]).reset_index(drop=True)


def score_attempts(frames: pd.DataFrame, attempts: pd.DataFrame) -> pd.DataFrame:
    """Add raw pitch control at the target, pass length, nearest-defender distance, and time."""
    p = default_params()
    gk = goalkeepers(frames).set_index(["period", "team"])["gk_id"]
    indexed = frames.set_index(["period", "frame"]).sort_index()
    times = indexed["t"].groupby(level=[0, 1]).first()
    pcs, lengths, nearest = [], [], []
    for a in attempts.itertuples():
        rows = indexed.loc[(a.period, a.frame)].reset_index()
        gks = {team: gk.get((a.period, team)) for team in ("home", "away")}
        scene = build_scene(rows, a.period, a.frame, a.team, gks,
                            ball_xy=np.array([a.start_x, a.start_y]), p=p)  # fmt: skip
        target = to_scene_coords(np.array([a.end_x, a.end_y]), a.team)
        pcs.append(pitch_control_at(target, scene, p))
        lengths.append(float(np.hypot(a.end_x - a.start_x, a.end_y - a.start_y)))
        d = np.linalg.norm(scene.def_pos - target, axis=1)
        nearest.append(float(d.min()) if len(d) else np.nan)
    out = attempts.copy()
    out["pitch_control"] = pcs
    out["length"] = lengths
    out["nearest_defender"] = nearest
    out["t"] = times.reindex(pd.MultiIndex.from_arrays([out["period"], out["frame"]])).to_numpy()
    return out
