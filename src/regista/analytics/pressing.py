"""Pressing: how close defenders are to the ball carrier, per frame and per window.

Thresholds are fixed a priori and never tuned (see docs/DESIGN.md, ADR-007):
- pressure: a defender within 5 yd (4.572 m) of the carrier, following
  StatsBomb's pressure definition (a fixed radius here; StatsBomb's grows
  towards the defending team's goal),
- tight pressure: a defender within 2 m.

The carrier is the frame-level ball owner. Pitch thirds are named from the
pressing team's point of view: "attacking" is a high press near the
opponent's goal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regista.schema import PITCH_LENGTH_M, Team

PRESSURE_RADIUS_M = 4.572  # 5 yards
TIGHT_PRESSURE_M = 2.0
THIRDS = ("defensive", "middle", "attacking")
FRAME_KEYS = ["match_id", "period", "frame"]


def pressure_frames(frames: pd.DataFrame, owner: pd.DataFrame) -> pd.DataFrame:
    """Per frame with a ball carrier: nearest-defender distance and defenders within 5 yd.

    ``owner`` is ``possession.ball_owner`` output. Columns: match_id, period,
    frame, t, carrier_id, carrier_team, pressing_team, nearest_defender_m,
    defenders_within, third.
    """
    carriers = owner.dropna(subset=["owner_id"])[[*FRAME_KEYS, "owner_id", "owner_team"]]
    players = frames.loc[frames["team"] != Team.BALL.value, [*FRAME_KEYS, "t", "team",
                                                             "player_id", "x", "y"]]  # fmt: skip
    c = carriers.merge(
        players.rename(columns={"player_id": "owner_id", "team": "owner_team"}),
        on=[*FRAME_KEYS, "owner_id", "owner_team"],
    )
    pairs = c.merge(players, on=FRAME_KEYS, suffixes=("", "_def"))
    pairs = pairs[pairs["team"] != pairs["owner_team"]]
    pairs["d"] = np.hypot(pairs["x_def"] - pairs["x"], pairs["y_def"] - pairs["y"])
    g = pairs.groupby([*FRAME_KEYS, "owner_id", "owner_team", "team", "t", "x"], sort=False)
    out = g["d"].agg(nearest_defender_m="min",
                     defenders_within=lambda d: int((d <= PRESSURE_RADIUS_M).sum()))  # fmt: skip
    out = out.reset_index().rename(
        columns={"owner_id": "carrier_id", "owner_team": "carrier_team", "team": "pressing_team"}
    )
    # carrier x in the pressing team's attacking frame (home attacks +x)
    x_att = np.where(out["pressing_team"] == Team.HOME.value, out["x"], -out["x"])
    edge = PITCH_LENGTH_M / 6
    out["third"] = np.select([x_att > edge, x_att < -edge], ["attacking", "defensive"], "middle")
    cols = [*FRAME_KEYS, "t", "carrier_id", "carrier_team", "pressing_team",
            "nearest_defender_m", "defenders_within", "third"]  # fmt: skip
    return out[cols].sort_values(FRAME_KEYS).reset_index(drop=True)


def press_windows(
    pressure: pd.DataFrame, window_s: float = 300.0, step_s: float | None = None
) -> pd.DataFrame:
    """Press intensity per pressing team, period, and trailing window, overall and by third.

    Windows end every ``step_s`` seconds (default: ``window_s``, i.e. tiling) and
    cover the preceding ``window_s`` seconds of the period; windows that would
    start before the period are cut at its start and flagged ``complete=False``.
    ``press_intensity`` is the share of opponent-carrier frames with a defender
    within 5 yd; ``tight_intensity`` uses 2 m.
    """
    step_s = window_s if step_s is None else step_s
    rows = []
    for (match_id, period, team), g in pressure.groupby(["match_id", "period", "pressing_team"]):
        t = g["t"].to_numpy()
        n_ends = int(np.floor(t.max() / step_s)) + 1
        for k in range(1, n_ends + 1):
            t_end = k * step_s
            t_start = max(t_end - window_s, 0.0)
            sel = g[(t >= t_start) & (t < t_end)]
            for third in ("all", *THIRDS):
                s = sel if third == "all" else sel[sel["third"] == third]
                n = len(s)
                rows.append(
                    {
                        "match_id": match_id,
                        "period": period,
                        "pressing_team": team,
                        "t_start": t_start,
                        "t_end": t_end,
                        "complete": t_end - window_s >= 0,
                        "third": third,
                        "n_frames": n,
                        "press_intensity": (s["nearest_defender_m"] <= PRESSURE_RADIUS_M).mean()
                        if n
                        else np.nan,
                        "tight_intensity": (s["nearest_defender_m"] <= TIGHT_PRESSURE_M).mean()
                        if n
                        else np.nan,
                        "mean_defenders_within": s["defenders_within"].mean() if n else np.nan,
                    }
                )
    return pd.DataFrame(rows)
