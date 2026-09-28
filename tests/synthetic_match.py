"""Synthetic matches for tests. Everything here is synthetic, by design, and labelled as such.

Two teams hold template shapes with gentle random motion; the ball is passed
between teammates at 15 m/s with holds in between, and possession changes after
a few passes. Hooks let tests change a team's formation, pressing distance, or
line height over time, so detectors can be checked against known answers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from regista.analytics.formations import TEMPLATES
from regista.analytics.kinematics import add_velocities
from regista.schema import validate_frames

BALL_SPEED = 15.0
HOLD_S = 1.5


@dataclass
class MatchSpec:
    period_s: float = 120.0
    fps: float = 10.0
    seed: int = 0
    # time (s, since period start) -> template name, per team
    home_formation: Callable[[int, float], str] = lambda period, t: "4-4-2"
    away_formation: Callable[[int, float], str] = lambda period, t: "4-4-2"
    # time -> defensive line depth offset in metres (negative = deeper), per team
    home_line: Callable[[int, float], float] = lambda period, t: 0.0
    away_line: Callable[[int, float], float] = lambda period, t: 0.0
    # time -> distance (m) the nearest defender keeps from the carrier; None = no press
    home_press: Callable[[int, float], float | None] = lambda period, t: None
    away_press: Callable[[int, float], float | None] = lambda period, t: None
    passes_per_possession: tuple[int, int] = (3, 6)
    extra: dict = field(default_factory=dict)


def _shape(name: str, line_offset: float) -> np.ndarray:
    """Template slots in metres for a team attacking +x, own goal at x = -52.5."""
    xy = np.array([[s.x, s.y] for s in TEMPLATES[name]], dtype=float)
    xy[:, 0] = -40.0 + xy[:, 0] * 0.9 + line_offset  # back line ~12 m from goal
    xy[:, 1] = xy[:, 1] * 0.8
    return xy


def make_match(spec: MatchSpec | None = None) -> pd.DataFrame:
    """Canonical frames (with velocities) for a synthetic two-period match."""
    spec = spec or MatchSpec()
    rng = np.random.default_rng(spec.seed)
    dt = 1.0 / spec.fps
    n = int(spec.period_s * spec.fps)
    ids = {team: [f"{team}_{i}" for i in range(10)] for team in ("home", "away")}
    phase = {team: rng.uniform(0, 2 * np.pi, (10, 2)) for team in ("home", "away")}
    rows = []
    frame0 = 0
    for period in (1, 2):
        # ball script: (carrier team, carrier index) holds then passes
        team, carrier = "home", 5
        ball = None
        state = "hold"
        t_state = 0.0
        passes_left = int(rng.integers(*spec.passes_per_possession))
        target = None
        for k in range(n):
            t = k * dt
            pos = {}
            for side in ("home", "away"):
                name = (spec.home_formation if side == "home" else spec.away_formation)(period, t)
                line = (spec.home_line if side == "home" else spec.away_line)(period, t)
                xy = _shape(name, line)
                wobble = np.column_stack(
                    [np.sin(0.3 * t + phase[side][:, 0]), np.sin(0.25 * t + phase[side][:, 1])]
                )
                xy = xy + wobble
                pos[side] = xy if side == "home" else -xy  # away attacks -x
            # pressing: nearest defender moves to a fixed distance from the carrier
            carrier_xy = pos[team][carrier]
            press = (spec.away_press if team == "home" else spec.home_press)(period, t)
            defending = "away" if team == "home" else "home"
            if press is not None:
                d = np.linalg.norm(pos[defending] - carrier_xy, axis=1)
                j = int(np.argmin(d))
                direction = (pos[defending][j] - carrier_xy) / max(d[j], 1e-6)
                pos[defending][j] = carrier_xy + direction * press

            if state == "hold":
                ball = carrier_xy + np.array([0.15, 0.0])
                if t - t_state >= HOLD_S:
                    passes_left -= 1
                    if passes_left <= 0:
                        new_team = defending
                        passes_left = int(rng.integers(*spec.passes_per_possession))
                    else:
                        new_team = team
                    options = [i for i in range(10) if not (new_team == team and i == carrier)]
                    target = (new_team, int(rng.choice(options)))
                    state, t_state, start = "fly", t, ball.copy()
            else:
                goal = pos[target[0]][target[1]]
                travelled = BALL_SPEED * (t - t_state)
                vec = goal - start
                dist = np.linalg.norm(vec)
                if travelled >= dist:
                    team, carrier = target
                    state, t_state = "hold", t
                    ball = pos[team][carrier] + np.array([0.15, 0.0])
                else:
                    ball = start + vec / dist * travelled

            f = frame0 + k
            for side in ("home", "away"):
                for i, (x, y) in enumerate(pos[side]):
                    rows.append((period, f, t, side, ids[side][i], x, y))
                gk_x = -50.0 if side == "home" else 50.0
                rows.append((period, f, t, side, f"{side}_gk", gk_x, 0.0))
            rows.append((period, f, t, "ball", "ball", ball[0], ball[1]))
        frame0 += n
    df = pd.DataFrame(rows, columns=["period", "frame", "t", "team", "player_id", "x", "y"])
    df["match_id"] = "synthetic"
    df["vx"] = np.nan
    df["vy"] = np.nan
    df["confidence"] = 1.0
    df["source"] = "metrica"
    return validate_frames(add_velocities(validate_frames(df)))
