"""Source-independent conversion steps shared by the ingest adapters.

kloppy datasets arrive in its default coordinate system: normalised [0, 1] on both
axes, origin top-left, y pointing down. These helpers turn that into canonical
frames (metres, centre origin, +y up, home attacking +x in every period).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regista.schema import (
    BALL_ID,
    BOUNDS_MARGIN_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    Source,
    Team,
    coerce_frames,
)

MIN_GK_PRESENCE = 0.5  # share of a period's frames a goalkeeper candidate must appear in
MIN_GK_DEPTH_M = 20.0  # goalkeepers closer to halfway than this make direction ambiguous


def to_metres(
    x_norm: np.ndarray | pd.Series,
    y_norm: np.ndarray | pd.Series,
    length: float = PITCH_LENGTH_M,
    width: float = PITCH_WIDTH_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Map normalised top-left/y-down coordinates to centre-origin metres with +y up."""
    x = (np.asarray(x_norm, dtype=float) - 0.5) * length
    y = (0.5 - np.asarray(y_norm, dtype=float)) * width
    return x, y


def wide_to_long(
    wide: pd.DataFrame, player_teams: dict[str, Team], match_id: str, source: Source
) -> pd.DataFrame:
    """Convert a kloppy ``to_df()`` table (one column pair per object) to canonical rows.

    Rows where an object has no position are dropped: absent objects have no row.
    Coordinates are converted to metres but direction is NOT normalised here.
    """
    base = pd.DataFrame(
        {
            "period": wide["period_id"].to_numpy(),
            "frame": wide["frame_id"].to_numpy(),
            "t": pd.to_timedelta(wide["timestamp"]).dt.total_seconds().to_numpy(),
        }
    )
    objects = {BALL_ID: Team.BALL, **player_teams}
    blocks = []
    for obj_id, team in objects.items():
        xcol, ycol = f"{obj_id}_x", f"{obj_id}_y"
        if xcol not in wide.columns:
            raise KeyError(f"no tracking columns for object {obj_id!r}")
        x, y = to_metres(wide[xcol].astype(float), wide[ycol].astype(float))
        block = base.assign(team=team.value, player_id=obj_id, x=x, y=y)
        blocks.append(block[np.isfinite(x) & np.isfinite(y)])
    frames = pd.concat(blocks, ignore_index=True)
    frames["match_id"] = match_id
    frames["vx"] = np.nan
    frames["vy"] = np.nan
    frames["confidence"] = 1.0
    frames["source"] = source.value
    return coerce_frames(frames)


def _goalkeeper_x(team_rows: pd.DataFrame) -> float:
    """Mean x of the team's deepest regular player (its goalkeeper) over a period."""
    n_frames = team_rows["frame"].nunique()
    presence = team_rows.groupby("player_id")["frame"].nunique() / n_frames
    mean_x = team_rows.groupby("player_id")["x"].mean()[presence >= MIN_GK_PRESENCE]
    return float(mean_x.loc[mean_x.abs().idxmax()])


def home_attack_flips(frames: pd.DataFrame) -> dict[int, bool]:
    """Per period, True if the home team attacks -x and must be flipped.

    Inferred from the data: each team's goalkeeper (its deepest regular player)
    stays in the half it defends. Kick-off positions are not used because
    broadcast tracking extrapolates some players across the halfway line there.
    Raises if the two goalkeepers are not clearly on opposite sides.
    """
    flips: dict[int, bool] = {}
    for period, g in frames.groupby("period"):
        home_gk = _goalkeeper_x(g[g["team"] == Team.HOME.value])
        away_gk = _goalkeeper_x(g[g["team"] == Team.AWAY.value])
        clear = min(abs(home_gk), abs(away_gk)) >= MIN_GK_DEPTH_M
        if not clear or np.sign(home_gk) == np.sign(away_gk):
            raise ValueError(
                f"period {period}: cannot infer attacking direction "
                f"(goalkeeper mean x: home {home_gk:.1f} m, away {away_gk:.1f} m)"
            )
        flips[int(period)] = home_gk > 0
    return flips


def normalise_direction(frames: pd.DataFrame, flips: dict[int, bool]) -> pd.DataFrame:
    """Rotate flipped periods by 180 degrees (negate x, y, vx, vy) so home attacks +x."""
    out = frames.copy()
    flip_periods = [p for p, f in flips.items() if f]
    mask = out["period"].isin(flip_periods)
    for col in ("x", "y", "vx", "vy"):
        out.loc[mask, col] = -out.loc[mask, col]
    return out


def drop_out_of_bounds(frames: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows beyond the schema's pitch bounds; return counts split by ball/player rows.

    Providers occasionally record real positions a few metres past the margin
    (a ball behind the goal after a shot). The schema margin is kept strict
    because it also filters off-pitch detections from CV. Callers must report
    the counts.
    """
    x_lim = PITCH_LENGTH_M / 2 + BOUNDS_MARGIN_M
    y_lim = PITCH_WIDTH_M / 2 + BOUNDS_MARGIN_M
    out = (frames["x"].abs() > x_lim) | (frames["y"].abs() > y_lim)
    is_ball = frames["team"] == Team.BALL.value
    counts = {"ball": int((out & is_ball).sum()), "player": int((out & ~is_ball).sum())}
    return frames[~out].reset_index(drop=True), counts
