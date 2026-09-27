"""Ball possession and pass inference from tracking data alone.

Owner: the nearest player within ``radius_m`` of the ball, provided the ball is
not moving faster than ``max_ball_speed`` (a ball in flight belongs to no one).
Passes: a change of owner between teammates. A change to an opponent is a turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regista.schema import Team

FRAME_KEYS = ["match_id", "period", "frame"]

PASS_COLUMNS = [
    "match_id",
    "period",
    "kind",
    "team",
    "from_player",
    "to_player",
    "start_frame",
    "end_frame",
]


def ball_owner(
    frames: pd.DataFrame, radius_m: float, max_ball_speed: float | None = None
) -> pd.DataFrame:
    """One row per frame with a ball position: nearest player, distance, and owner.

    ``owner_id``/``owner_team`` are NA when nobody is within ``radius_m`` or the
    ball is faster than ``max_ball_speed`` (which needs ball ``vx``/``vy``; a NaN
    ball speed fails the gate). Frames without a ball row are absent.
    """
    is_ball = frames["team"] == Team.BALL.value
    ball = frames.loc[is_ball, [*FRAME_KEYS, "x", "y", "vx", "vy"]]
    players = frames.loc[~is_ball, [*FRAME_KEYS, "team", "player_id", "x", "y"]]
    pairs = players.merge(ball, on=FRAME_KEYS, suffixes=("", "_ball"))
    pairs["dist"] = np.hypot(pairs["x"] - pairs["x_ball"], pairs["y"] - pairs["y_ball"])
    nearest = pairs.sort_values("dist", kind="stable").drop_duplicates(FRAME_KEYS)

    owned = nearest["dist"] <= radius_m
    if max_ball_speed is not None:
        ball_speed = np.hypot(nearest["vx"], nearest["vy"])
        owned &= ball_speed <= max_ball_speed

    out = nearest[FRAME_KEYS].copy()
    out["nearest_id"] = nearest["player_id"]
    out["dist"] = nearest["dist"]
    out["owner_id"] = nearest["player_id"].where(owned)
    out["owner_team"] = nearest["team"].where(owned)
    return out.sort_values(FRAME_KEYS).reset_index(drop=True)


def possession_spells(owner: pd.DataFrame, min_hold_frames: int = 1) -> pd.DataFrame:
    """Collapse frame-level ownership into spells of one player holding the ball.

    Unowned frames between two frames of the same owner do not break a spell.
    Spells shorter than ``min_hold_frames`` are dropped, and neighbours with the
    same owner are then merged.
    """
    o = owner.dropna(subset=["owner_id"]).sort_values(FRAME_KEYS)

    def runs(df: pd.DataFrame, weight: pd.Series) -> pd.DataFrame:
        new = (
            (df["owner_id"] != df["owner_id"].shift())
            | (df["period"] != df["period"].shift())
            | (df["match_id"] != df["match_id"].shift())
        )
        rid = np.cumsum(new.fillna(True).to_numpy(dtype=bool))
        g = df.assign(_w=weight.to_numpy()).groupby(rid, sort=False)
        return pd.DataFrame(
            {
                "match_id": g["match_id"].first(),
                "period": g["period"].first(),
                "player_id": g["owner_id"].first(),
                "team": g["owner_team"].first(),
                "first_frame": g["first_frame"].min(),
                "last_frame": g["last_frame"].max(),
                "n_frames": g["_w"].sum(),
            }
        ).reset_index(drop=True)

    o = o.assign(first_frame=o["frame"], last_frame=o["frame"])
    spells = runs(o, pd.Series(1, index=o.index))
    spells = spells[spells["n_frames"] >= min_hold_frames]
    spells = spells.rename(columns={"player_id": "owner_id", "team": "owner_team"})
    merged = runs(spells, spells["n_frames"])
    return merged


def detect_passes(
    owner: pd.DataFrame, min_hold_frames: int = 1, max_gap_frames: int | None = None
) -> pd.DataFrame:
    """Passes and turnovers from consecutive possession spells within a period.

    ``start_frame`` is the passer's last frame on the ball, ``end_frame`` the
    receiver's first. Transitions separated by more than ``max_gap_frames`` of
    unowned ball (e.g. the ball going out of play) are not counted.
    """
    s = possession_spells(owner, min_hold_frames)
    nxt = s.shift(-1)
    same_period = (nxt["match_id"] == s["match_id"]) & (nxt["period"] == s["period"])
    keep = same_period.fillna(False).to_numpy(dtype=bool)
    if max_gap_frames is not None:
        keep = keep & (nxt["first_frame"] - s["last_frame"]).le(max_gap_frames).to_numpy(bool)
    same_team = (nxt["team"] == s["team"]).fillna(False).to_numpy(dtype=bool)
    passes = pd.DataFrame(
        {
            "match_id": s["match_id"],
            "period": s["period"],
            "kind": np.where(same_team, "pass", "turnover"),
            "team": s["team"],
            "from_player": s["player_id"],
            "to_player": nxt["player_id"],
            "start_frame": s["last_frame"],
            "end_frame": nxt["first_frame"],
        }
    )[keep]
    return passes.astype({"start_frame": "int64", "end_frame": "int64"}).reset_index(drop=True)


@dataclass
class PassScore:
    tp: int
    fp: int
    fn: int
    matched_pred: pd.Series  # bool per predicted pass
    matched_true: pd.Series  # bool per true pass

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score_passes(pred: pd.DataFrame, truth: pd.DataFrame, tol_frames: int) -> PassScore:
    """One-to-one matching of predicted to true passes.

    A match needs the same period, passer, and receiver, with start frames at most
    ``tol_frames`` apart. Pairs are matched greedily, closest in time first.
    Both tables need ``period, from_player, to_player, start_frame``.
    """
    p = pred.reset_index(drop=True).rename_axis("pi").reset_index()
    t = truth.reset_index(drop=True).rename_axis("ti").reset_index()
    cand = p.merge(t, on=["period", "from_player", "to_player"], suffixes=("_p", "_t"))
    cand["dt"] = (cand["start_frame_p"] - cand["start_frame_t"]).abs()
    cand = cand[cand["dt"] <= tol_frames].sort_values(["dt", "pi", "ti"], kind="stable")

    used_p: set[int] = set()
    used_t: set[int] = set()
    for pi, ti in zip(cand["pi"], cand["ti"], strict=True):
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)

    matched_pred = p["pi"].isin(used_p)
    matched_true = t["ti"].isin(used_t)
    return PassScore(
        tp=len(used_p),
        fp=int((~matched_pred).sum()),
        fn=int((~matched_true).sum()),
        matched_pred=matched_pred,
        matched_true=matched_true,
    )
