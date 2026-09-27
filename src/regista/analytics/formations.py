"""Formation and role detection per team, time window, and phase (in / out of possession).

For each window, each team's 10 outfield players get a mean position relative to
the team centroid. Positions are expressed in the team's attacking frame (+x
towards the opponent goal, +y to the left) and scaled per axis to unit spread,
then matched to template formations with the Hungarian algorithm
(``scipy.optimize.linear_sum_assignment``). The template with the lowest cost
wins; confidence is the relative cost margin over the runner-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from regista.schema import Team

N_OUTFIELD = 10
MIN_PRESENCE = 0.5  # a player must appear in this share of a window's frames


@dataclass(frozen=True)
class Slot:
    role: str
    group: str  # DEF / MID / FWD
    x: float  # depth, towards the opponent goal
    y: float  # width, + is the team's left


def _line(group: str, x: float, roles_ys: list[tuple[str, float]]) -> list[Slot]:
    return [Slot(role, group, x, y) for role, y in roles_ys]


BACK4 = _line("DEF", 0, [("LB", 24), ("LCB", 8), ("RCB", -8), ("RB", -24)])
BACK3 = _line("DEF", 0, [("LCB", 16), ("CB", 0), ("RCB", -16)])

TEMPLATES: dict[str, list[Slot]] = {
    "4-4-2": BACK4
    + _line("MID", 20, [("LM", 24), ("LCM", 8), ("RCM", -8), ("RM", -24)])
    + _line("FWD", 38, [("LS", 8), ("RS", -8)]),
    "4-3-3": BACK4
    + _line("MID", 20, [("LCM", 14), ("CM", 0), ("RCM", -14)])
    + _line("FWD", 38, [("LW", 24), ("ST", 0), ("RW", -24)]),
    "4-2-3-1": BACK4
    + _line("MID", 14, [("LDM", 8), ("RDM", -8)])
    + _line("MID", 28, [("LAM", 22), ("CAM", 0), ("RAM", -22)])
    + _line("FWD", 40, [("ST", 0)]),
    "4-1-4-1": BACK4
    + _line("MID", 12, [("DM", 0)])
    + _line("MID", 24, [("LM", 24), ("LCM", 8), ("RCM", -8), ("RM", -24)])
    + _line("FWD", 40, [("ST", 0)]),
    "3-5-2": BACK3
    + _line("MID", 16, [("LWB", 28), ("RWB", -28)])
    + _line("MID", 20, [("LCM", 12), ("CM", 0), ("RCM", -12)])
    + _line("FWD", 38, [("LS", 8), ("RS", -8)]),
    "3-4-3": BACK3
    + _line("MID", 20, [("LM", 26), ("LCM", 9), ("RCM", -9), ("RM", -26)])
    + _line("FWD", 38, [("LW", 20), ("ST", 0), ("RW", -20)]),
    "5-3-2": _line("DEF", 4, [("LWB", 28), ("RWB", -28)])
    + BACK3
    + _line("MID", 20, [("LCM", 14), ("CM", 0), ("RCM", -14)])
    + _line("FWD", 38, [("LS", 8), ("RS", -8)]),
}


def unit_shape(xy: np.ndarray) -> np.ndarray:
    """Centre on the centroid and scale each axis to unit standard deviation."""
    centred = xy - xy.mean(axis=0)
    std = centred.std(axis=0)
    if (std <= 0).any():
        raise ValueError("degenerate shape: no spread along one axis")
    return centred / std


_TEMPLATE_SHAPES = {
    name: unit_shape(np.array([[s.x, s.y] for s in slots])) for name, slots in TEMPLATES.items()
}


@dataclass
class ShapeMatch:
    label: str
    cost: float  # mean squared distance to the matched template slots (unit shape)
    runner_up: str  # second-best template
    runner_up_cost: float
    slots: list[Slot]  # matched slot per input player, in input order

    @property
    def margin(self) -> float:
        """Absolute cost gap to the runner-up (unit-shape squared distance)."""
        return self.runner_up_cost - self.cost

    @property
    def confidence(self) -> float:
        """Relative cost margin over the runner-up, in [0, 1]."""
        return self.margin / self.runner_up_cost if self.runner_up_cost > 0 else 0.0


def _assign(shape: np.ndarray) -> dict[str, tuple[float, list[int]]]:
    """Per template: (mean squared distance, slot index per player) after optimal assignment."""
    out = {}
    for name, template in _TEMPLATE_SHAPES.items():
        cost = ((shape[:, None, :] - template[None, :, :]) ** 2).sum(axis=2)
        rows, cols = linear_sum_assignment(cost)
        slot_of = dict(zip(rows, cols, strict=True))
        out[name] = (float(cost[rows, cols].mean()), [slot_of[i] for i in range(N_OUTFIELD)])
    return out


def _check(xy: np.ndarray) -> np.ndarray:
    if xy.shape != (N_OUTFIELD, 2):
        raise ValueError(f"expected {N_OUTFIELD} outfield positions, got shape {xy.shape}")
    return unit_shape(xy)


def template_costs(xy: np.ndarray) -> dict[str, float]:
    """Assignment cost of 10 outfield positions (attacking frame) against every template."""
    return {name: cost for name, (cost, _) in _assign(_check(xy)).items()}


def classify_shape(xy: np.ndarray) -> ShapeMatch:
    """Match 10 outfield positions (attacking frame) to the best template."""
    ranked = sorted(_assign(_check(xy)).items(), key=lambda kv: kv[1][0])
    (label, (best, cols)), (runner_up, (second, _)) = ranked[0], ranked[1]
    return ShapeMatch(label, best, runner_up, second, [TEMPLATES[label][c] for c in cols])


def to_attacking_frame(frames: pd.DataFrame) -> pd.DataFrame:
    """Rotate away-team positions 180 degrees so every team attacks +x, left is +y."""
    out = frames.copy()
    away = (out["team"] == Team.AWAY.value).to_numpy()
    out.loc[away, "x"] = -out.loc[away, "x"]
    out.loc[away, "y"] = -out.loc[away, "y"]
    return out


def window_shapes(
    frames: pd.DataFrame, possession: pd.DataFrame, window_s: float, step_s: float
) -> pd.DataFrame:
    """Mean centroid-relative position per outfield player, per team/period/window/phase.

    ``possession`` is a per-frame team phase (``team_in_possession``). The phase
    is "in" when the player's team has the ball and "out" when the opponent
    does; frames with no phase are ignored. The goalkeeper is the player deepest
    on average in the window; the 10 most-present other players (each in at
    least ``MIN_PRESENCE`` of the frames) form the shape.
    """
    keys = ["match_id", "period", "frame"]
    players = frames[frames["team"] != Team.BALL.value]
    players = to_attacking_frame(players).merge(
        possession.rename(columns={"team": "possession_team"}), on=keys
    )
    players = players.dropna(subset=["possession_team"])
    players["phase"] = np.where(players["possession_team"] == players["team"], "in", "out")
    players["window"] = (players["t"] // step_s).astype(int)
    span = max(int(round(window_s / step_s)), 1)

    rows = []
    for (match_id, period, team, phase), g in players.groupby(
        ["match_id", "period", "team", "phase"]
    ):
        for w in range(int(g["window"].min()), int(g["window"].max()) - span + 2):
            win = g[g["window"].between(w, w + span - 1)]
            n_frames = win["frame"].nunique()
            presence = win.groupby("player_id")["frame"].nunique() / n_frames
            depth = win.groupby("player_id")["x"].mean()
            regulars = presence[presence >= MIN_PRESENCE].index
            if len(regulars) == 0:
                continue
            gk = depth.loc[regulars].idxmin()
            outfield = presence.drop(gk).loc[lambda s: s >= MIN_PRESENCE]
            outfield = outfield.sort_values(ascending=False).index[:N_OUTFIELD]
            sel = win[win["player_id"].isin(outfield)]
            centroid = sel.groupby("frame")[["x", "y"]].transform("mean")
            rel = sel[["player_id"]].assign(
                rx=sel["x"] - centroid["x"], ry=sel["y"] - centroid["y"]
            )
            mean = rel.groupby("player_id")[["rx", "ry"]].mean()
            for pid, r in mean.iterrows():
                rows.append(
                    {
                        "match_id": match_id,
                        "period": period,
                        "team": team,
                        "phase": phase,
                        "window": w,
                        "t_start": w * step_s,
                        "t_end": (w + span) * step_s,
                        "n_frames": n_frames,
                        "n_outfield": len(outfield),
                        "player_id": pid,
                        "gk_id": gk,
                        "x": r["rx"],
                        "y": r["ry"],
                    }
                )
    return pd.DataFrame(rows)


def detect_formations(shapes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify each window's shape.

    Returns (formations, roles). ``formations`` has one row per
    team/period/phase/window with ``label``, ``cost``, ``runner_up``,
    ``runner_up_cost``, ``margin``, ``confidence`` and ``cost_<template>`` for
    every template; windows without exactly
    10 outfield players get label NA. ``roles`` has one row per player per
    window with ``role`` and ``group``.
    """
    keys = ["match_id", "period", "team", "phase", "window"]
    form_rows, role_rows = [], []
    for key, g in shapes.groupby(keys, sort=True):
        base = dict(zip(keys, key, strict=True))
        base.update(t_start=g["t_start"].iat[0], t_end=g["t_end"].iat[0])
        if len(g) != N_OUTFIELD:
            form_rows.append({**base, "label": None, "runner_up": None})
            continue
        xy = g[["x", "y"]].to_numpy()
        match = classify_shape(xy)
        costs = {f"cost_{name}": c for name, c in template_costs(xy).items()}
        form_rows.append(
            {
                **base,
                **costs,
                "label": match.label,
                "cost": match.cost,
                "runner_up": match.runner_up,
                "runner_up_cost": match.runner_up_cost,
                "margin": match.margin,
                "confidence": match.confidence,
            }
        )
        for pid, slot in zip(g["player_id"], match.slots, strict=True):
            role_rows.append({**base, "player_id": pid, "role": slot.role, "group": slot.group})
    formations = pd.DataFrame(form_rows).astype({"label": "string", "runner_up": "string"})
    return formations, pd.DataFrame(role_rows)


def change_points(formations: pd.DataFrame, min_windows: int = 2) -> pd.DataFrame:
    """Windows where a team's formation changes and the new label holds.

    Per match/team/phase, windows are taken in time order across periods. A
    change is flagged at the first window of a run of at least ``min_windows``
    consecutive windows whose label differs from the current established label.
    Windows with no label are skipped.
    """
    rows = []
    order = ["match_id", "team", "phase"]
    for key, g in formations.dropna(subset=["label"]).groupby(order):
        g = g.sort_values(["period", "window"]).reset_index(drop=True)
        labels = g["label"].tolist()
        current = labels[0]
        i = 1
        while i < len(labels):
            run = labels[i : i + min_windows]
            if labels[i] != current and len(run) == min_windows and len(set(run)) == 1:
                rows.append(
                    {
                        **dict(zip(order, key, strict=True)),
                        "period": g.loc[i, "period"],
                        "t_start": g.loc[i, "t_start"],
                        "from_label": current,
                        "to_label": labels[i],
                    }
                )
                current = labels[i]
                i += min_windows
            else:
                i += 1
    return pd.DataFrame(rows, columns=[*order, "period", "t_start", "from_label", "to_label"])


def label_stability(formations: pd.DataFrame) -> pd.DataFrame:
    """Share of consecutive labelled windows with an unchanged label (label-free metric).

    Per match/team/phase, windows are ordered in time across periods; windows
    without a label are skipped. Columns: pairs, unchanged, stability.
    """
    rows = []
    order = ["match_id", "team", "phase"]
    for key, g in formations.dropna(subset=["label"]).groupby(order):
        labels = g.sort_values(["period", "window"])["label"].to_numpy()
        pairs = len(labels) - 1
        unchanged = int((labels[1:] == labels[:-1]).sum()) if pairs > 0 else 0
        rows.append(
            {
                **dict(zip(order, key, strict=True)),
                "pairs": pairs,
                "unchanged": unchanged,
                "stability": unchanged / pairs if pairs > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=[*order, "pairs", "unchanged", "stability"])


def role_consistency(roles: pd.DataFrame) -> pd.DataFrame:
    """Per player and phase: share of windows spent in their most frequent role and group.

    Columns: windows, modal_role, role_consistency, modal_group, group_consistency.
    """
    keys = ["match_id", "team", "phase", "player_id"]
    g = roles.groupby(keys)
    out = pd.DataFrame(
        {
            "windows": g.size(),
            "modal_role": g["role"].agg(lambda s: s.value_counts().index[0]),
            "role_consistency": g["role"].agg(lambda s: s.value_counts().iloc[0] / len(s)),
            "modal_group": g["group"].agg(lambda s: s.value_counts().index[0]),
            "group_consistency": g["group"].agg(lambda s: s.value_counts().iloc[0] / len(s)),
        }
    )
    return out.reset_index()
