# Adapted from Friends-of-Tracking-Data-FoTD/LaurieOnTracking/Metrica_PitchControl.py (MIT)
# Copyright (c) 2021 Friends-of-Tracking-Data-FoTD. See THIRD_PARTY_NOTICES.md.
"""Passing options from a pitch-control model (Spearman 2018, LaurieOnTracking parameters).

Pitch control at a point is the modelled probability that the attacking team
gains control if the ball is played there now. Each player's time to intercept
assumes they keep their current velocity for ``reaction_time`` seconds, then run
straight at ``max_player_speed``. Control accumulates over time once the ball
arrives (travelling at ``average_ball_speed``), weighted by a sigmoid arrival
probability and a control rate ``lambda``.

Differences from the original: players are handled as arrays rather than
objects, inputs come from canonical frames in the attacking team's frame (+x
towards the opponent goal), and a failure to converge raises instead of printing.

"Probability" here is a model output. How well it is calibrated is measured in
``eval/passing_options.py``, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from regista.schema import Team


def default_params(time_to_control_veto: float = 3.0) -> dict[str, float]:
    """LaurieOnTracking's published defaults (``default_model_params``)."""
    p = {
        "max_player_speed": 5.0,  # m/s
        "reaction_time": 0.7,  # s
        "tti_sigma": 0.45,  # s, arrival-time uncertainty
        "kappa_def": 1.0,
        "lambda_att": 4.3,  # 1/s, ball control rate
        "average_ball_speed": 15.0,  # m/s
        "int_dt": 0.04,  # s
        "max_int_time": 10.0,  # s
        "model_converge_tol": 0.01,
        "offside_tol": 0.2,  # m
    }
    p["lambda_def"] = p["lambda_att"] * p["kappa_def"]
    p["lambda_gk"] = p["lambda_def"] * 3.0
    spread = np.sqrt(3) * p["tti_sigma"] / np.pi
    p["time_to_control_att"] = time_to_control_veto * np.log(10) * (spread + 1 / p["lambda_att"])
    p["time_to_control_def"] = time_to_control_veto * np.log(10) * (spread + 1 / p["lambda_def"])
    return p


@dataclass
class Scene:
    """One instant, in the attacking team's frame (+x towards the opponent goal)."""

    att_ids: np.ndarray
    att_pos: np.ndarray  # (n, 2)
    att_vel: np.ndarray  # (n, 2); NaN velocities treated as 0
    def_pos: np.ndarray  # (m, 2)
    def_vel: np.ndarray
    def_is_gk: np.ndarray  # (m,) bool
    ball: np.ndarray  # (2,)
    offside_ids: list[str] = field(default_factory=list)


def remove_offside(scene: Scene, tol: float) -> Scene:
    """Drop attackers beyond the offside line (second-deepest defender, ball, or halfway)."""
    if len(scene.def_pos) < 2:
        return scene
    second_deepest = np.sort(scene.def_pos[:, 0])[-2]
    line = max(second_deepest, scene.ball[0], 0.0) + tol
    onside = scene.att_pos[:, 0] <= line
    return Scene(
        scene.att_ids[onside],
        scene.att_pos[onside],
        scene.att_vel[onside],
        scene.def_pos,
        scene.def_vel,
        scene.def_is_gk,
        scene.ball,
        offside_ids=[str(i) for i in scene.att_ids[~onside]],
    )


def _time_to_intercept(pos: np.ndarray, vel: np.ndarray, target: np.ndarray, p: dict) -> np.ndarray:
    r_reaction = pos + vel * p["reaction_time"]
    return p["reaction_time"] + np.linalg.norm(target - r_reaction, axis=1) / p["max_player_speed"]


def pitch_control_at(target: np.ndarray, scene: Scene, p: dict) -> float:
    """Attacking team's pitch control at one target point (Spearman eq. 3)."""
    ball_time = np.linalg.norm(target - scene.ball) / p["average_ball_speed"]
    tti_att = _time_to_intercept(scene.att_pos, scene.att_vel, target, p)
    tti_def = _time_to_intercept(scene.def_pos, scene.def_vel, target, p)
    if len(tti_att) == 0:
        return 0.0
    if len(tti_def) == 0:
        return 1.0
    tau_att, tau_def = tti_att.min(), tti_def.min()
    if tau_att - max(ball_time, tau_def) >= p["time_to_control_def"]:
        return 0.0
    if tau_def - max(ball_time, tau_att) >= p["time_to_control_att"]:
        return 1.0

    keep_att = tti_att - tau_att < p["time_to_control_att"]
    keep_def = tti_def - tau_def < p["time_to_control_def"]
    tti_att, tti_def = tti_att[keep_att], tti_def[keep_def]
    lam_att = np.full(len(tti_att), p["lambda_att"])
    lam_def = np.where(scene.def_is_gk[keep_def], p["lambda_gk"], p["lambda_def"])
    k = np.pi / np.sqrt(3.0) / p["tti_sigma"]
    dt = p["int_dt"]

    times = np.arange(ball_time - dt, ball_time + p["max_int_time"], dt)
    ppcf_att = np.zeros(len(tti_att))
    ppcf_def = np.zeros(len(tti_def))
    total_att = total_def = 0.0
    for t in times[1:]:
        remaining = 1.0 - total_att - total_def
        ppcf_att += remaining * lam_att / (1.0 + np.exp(-k * (t - tti_att))) * dt
        ppcf_def += remaining * lam_def / (1.0 + np.exp(-k * (t - tti_def))) * dt
        total_att, total_def = ppcf_att.sum(), ppcf_def.sum()
        if total_att + total_def >= 1.0 - p["model_converge_tol"]:
            return float(total_att)
    raise RuntimeError(f"pitch control did not converge (total {total_att + total_def:.3f})")


def to_scene_coords(xy: np.ndarray, attacking_team: str) -> np.ndarray:
    """Canonical coordinates -> the attacking team's frame (home already attacks +x)."""
    xy = np.asarray(xy, dtype=float)
    return -xy if attacking_team == Team.AWAY.value else xy


def build_scene(
    frames: pd.DataFrame,
    period: int,
    frame: int,
    attacking_team: str,
    goalkeepers: dict[str, str],
    ball_xy: np.ndarray | None = None,
    offside: bool = True,
    p: dict | None = None,
) -> Scene:
    """Scene at one frame of canonical frames, in ``attacking_team``'s frame.

    ``goalkeepers`` maps team -> goalkeeper id (e.g. from ``shape.goalkeepers``).
    ``ball_xy`` (canonical coordinates) overrides the tracked ball position.
    """
    p = p or default_params()
    rows = frames[(frames["period"] == period) & (frames["frame"] == frame)]
    sign = -1.0 if attacking_team == Team.AWAY.value else 1.0
    pos = rows[["x", "y"]].to_numpy(float) * sign
    vel = np.nan_to_num(rows[["vx", "vy"]].to_numpy(float)) * sign
    team = rows["team"].to_numpy()
    is_att = team == attacking_team
    is_def = (team != attacking_team) & (team != Team.BALL.value)
    if ball_xy is None:
        is_ball = team == Team.BALL.value
        if not is_ball.any():
            raise ValueError(f"no ball position at period {period} frame {frame}")
        ball = pos[is_ball][0]
    else:
        ball = to_scene_coords(ball_xy, attacking_team)
    defending_team = Team.HOME.value if attacking_team == Team.AWAY.value else Team.AWAY.value
    scene = Scene(
        att_ids=rows["player_id"].to_numpy()[is_att],
        att_pos=pos[is_att],
        att_vel=vel[is_att],
        def_pos=pos[is_def],
        def_vel=vel[is_def],
        def_is_gk=(rows["player_id"].to_numpy()[is_def] == goalkeepers.get(defending_team)),
        ball=ball,
    )
    return remove_offside(scene, p["offside_tol"]) if offside else scene


def passing_options(scene: Scene, p: dict | None = None) -> pd.DataFrame:
    """Pitch control at each onside teammate's current position (the passing options).

    The ball carrier (the attacker nearest the ball) is excluded. These values
    cannot be validated for teammates who were not passed to.
    """
    p = p or default_params()
    if len(scene.att_ids) == 0:
        return pd.DataFrame(columns=["player_id", "x", "y", "pitch_control"])
    carrier = np.argmin(np.linalg.norm(scene.att_pos - scene.ball, axis=1))
    rows = [
        {"player_id": pid, "x": xy[0], "y": xy[1], "pitch_control": pitch_control_at(xy, scene, p)}
        for i, (pid, xy) in enumerate(zip(scene.att_ids, scene.att_pos, strict=True))
        if i != carrier
    ]
    return pd.DataFrame(rows).sort_values("pitch_control", ascending=False).reset_index(drop=True)


def pitch_control_grid(
    scene: Scene, n_x: int = 50, length: float = 105.0, width: float = 68.0, p: dict | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full pitch-control surface (for plots only). Returns (surface[y, x], xgrid, ygrid)."""
    p = p or default_params()
    n_y = int(n_x * width / length)
    xgrid = (np.arange(n_x) + 0.5) * length / n_x - length / 2
    ygrid = (np.arange(n_y) + 0.5) * width / n_y - width / 2
    surface = np.array(
        [[pitch_control_at(np.array([x, y]), scene, p) for x in xgrid] for y in ygrid]
    )
    return surface, xgrid, ygrid
