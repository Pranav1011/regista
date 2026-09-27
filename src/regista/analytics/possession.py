"""Ball possession and pass inference from tracking data alone.

Owner: the nearest player within ``radius_m`` of the ball, provided the ball is
not moving faster than ``max_ball_speed`` (a ball in flight belongs to no one).
Passes: a change of owner between teammates. A change to an opponent is a turnover.

Two pass detectors are kept side by side so both stay reproducible:
- v1 (``detect_passes``): fixed radius; the pass starts at the passer's last touch.
- v2 (``detect_passes_v2``): per-match radius from unlabelled data
  (``match_radius``), and a teammate transition only counts as a pass if the
  ball accelerates away from the passer (``release_speed`` / ``release_cos``);
  the pass starts at that release frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regista.analytics.kinematics import frame_interval
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
    """One row per frame in ``frames``: nearest player, distance, and owner.

    ``owner_id``/``owner_team`` are NA when nobody is within ``radius_m``, the
    ball is faster than ``max_ball_speed`` (which needs ball ``vx``/``vy``; a NaN
    ball speed fails the gate), or the frame has no ball row. An owner is never
    carried forward from an earlier frame.
    """
    is_ball = frames["team"] == Team.BALL.value
    ball = frames.loc[is_ball, [*FRAME_KEYS, "x", "y", "vx", "vy"]]
    players = frames.loc[~is_ball, [*FRAME_KEYS, "team", "player_id", "x", "y"]]
    pairs = players.merge(ball, on=FRAME_KEYS, suffixes=("", "_ball"))
    pairs["dist"] = np.hypot(pairs["x"] - pairs["x_ball"], pairs["y"] - pairs["y_ball"])
    nearest = pairs.sort_values("dist", kind="stable").drop_duplicates(FRAME_KEYS)

    ball_speed = np.hypot(nearest["vx"], nearest["vy"])
    owned = nearest["dist"] <= radius_m
    if max_ball_speed is not None:
        owned &= ball_speed <= max_ball_speed

    per_frame = nearest[FRAME_KEYS].copy()
    per_frame["nearest_id"] = nearest["player_id"]
    per_frame["dist"] = nearest["dist"]
    per_frame["ball_speed"] = ball_speed
    per_frame["owner_id"] = nearest["player_id"].where(owned)
    per_frame["owner_team"] = nearest["team"].where(owned)

    all_frames = frames[FRAME_KEYS].drop_duplicates()
    out = all_frames.merge(per_frame, on=FRAME_KEYS, how="left")
    out["ball_visible"] = (
        out.merge(ball[FRAME_KEYS].assign(_b=True), on=FRAME_KEYS, how="left")["_b"]
        .notna()
        .to_numpy()
    )
    return out.sort_values(FRAME_KEYS).reset_index(drop=True)


def possession_spells(owner: pd.DataFrame, min_hold_frames: int = 1) -> pd.DataFrame:
    """Collapse frame-level ownership into spells of one player holding the ball.

    Unowned frames (including frames without a ball) between two frames of the
    same owner do not break a spell; they are not assigned to anyone either.
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


def transitions(
    owner: pd.DataFrame, min_hold_frames: int = 1, max_gap_frames: int | None = None
) -> pd.DataFrame:
    """Consecutive possession spells within a period as passes (same team) or turnovers.

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


def detect_passes(
    owner: pd.DataFrame, min_hold_frames: int = 1, max_gap_frames: int | None = None
) -> pd.DataFrame:
    """v1: every transition between spells, passes starting at the passer's last touch."""
    return transitions(owner, min_hold_frames, max_gap_frames)


KICK_START_MS = 3.0  # ball speed crossing upwards through this marks a possible kick
KICK_PEAK_MS = 7.0  # ...confirmed if the ball reaches this speed
KICK_WINDOW_S = 0.4  # ...within this time


def kick_onset_distances(frames: pd.DataFrame) -> pd.Series:
    """Ball-to-nearest-player distance just before each kick, from unlabelled tracking.

    A kick is an upward crossing of ``KICK_START_MS`` between consecutive frames
    that reaches ``KICK_PEAK_MS`` within ``KICK_WINDOW_S`` in the same period.
    The distance is taken at the frame before the crossing, when the kicker is
    on the ball. Needs ball ``vx``/``vy``.
    """
    o = ball_owner(frames, radius_m=np.inf).sort_values(FRAME_KEYS).reset_index(drop=True)
    speed = o["ball_speed"].to_numpy(dtype=float)
    group = o.groupby(["match_id", "period"], sort=False).ngroup().to_numpy()
    contiguous = (group[1:] == group[:-1]) & (np.diff(o["frame"].to_numpy()) == 1)
    up = np.flatnonzero(contiguous & (speed[:-1] < KICK_START_MS) & (speed[1:] >= KICK_START_MS))
    up += 1
    horizon = max(int(round(KICK_WINDOW_S / frame_interval(frames))), 1)
    peak = np.array(
        [
            np.nanmax(np.where(group[i : i + horizon] == group[i], speed[i : i + horizon], np.nan))
            for i in up
        ]
    )
    return pd.Series(o["dist"].to_numpy()[up[peak >= KICK_PEAK_MS] - 1], name="dist")


def match_radius(frames: pd.DataFrame, quantile: float) -> float:
    """Ownership radius for one match, from unlabelled tracking only.

    The ``quantile`` of ball-to-nearest-player distances at kick onsets
    (``kick_onset_distances``): how far from the ball a player can be while
    still playing it, as recorded in this match.
    """
    if frames["match_id"].nunique() != 1:
        raise ValueError("match_radius needs frames from exactly one match")
    dists = kick_onset_distances(frames).dropna()
    if dists.empty:
        raise ValueError("no kick onsets found to derive an ownership radius from")
    return float(dists.quantile(quantile))


def release_features(
    frames: pd.DataFrame, passes: pd.DataFrame, back_frames: int, fwd_frames: int
) -> pd.DataFrame:
    """Ball speed and direction relative to the passer around each pass's last touch.

    One row per (pass index, frame) for frames from ``start_frame - back_frames``
    to ``min(start_frame + fwd_frames, end_frame)``. ``cos`` is the cosine between
    the ball velocity and the passer-to-ball direction (1 = straight away).
    """
    lo = passes["start_frame"].to_numpy() - back_frames
    hi = np.minimum(passes["start_frame"].to_numpy() + fwd_frames, passes["end_frame"].to_numpy())
    lengths = np.maximum(hi - lo + 1, 0)
    row = np.repeat(np.arange(len(passes)), lengths)  # window row -> pass position
    offsets = np.arange(len(row)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    window = pd.DataFrame(
        {
            "pass_idx": passes.index.to_numpy()[row],
            "match_id": passes["match_id"].to_numpy()[row],
            "period": passes["period"].to_numpy()[row],
            "frame": lo[row] + offsets,
            "player_id": passes["from_player"].to_numpy()[row],
        }
    )
    ball = frames.loc[frames["team"] == Team.BALL.value, [*FRAME_KEYS, "x", "y", "vx", "vy"]]
    players = frames.loc[frames["team"] != Team.BALL.value, [*FRAME_KEYS, "player_id", "x", "y"]]
    w = window.merge(ball, on=FRAME_KEYS).merge(
        players, on=[*FRAME_KEYS, "player_id"], suffixes=("_ball", "_passer")
    )
    dx, dy = w["x_ball"] - w["x_passer"], w["y_ball"] - w["y_passer"]
    speed = np.hypot(w["vx"], w["vy"])
    norm = np.hypot(dx, dy) * speed
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (w["vx"] * dx + w["vy"] * dy) / norm
    return pd.DataFrame(
        {"pass_idx": w["pass_idx"], "frame": w["frame"], "speed": speed, "cos": cos}
    )


def apply_release_gate(
    detected: pd.DataFrame, features: pd.DataFrame, release_speed: float, release_cos: float
) -> pd.DataFrame:
    """Keep passes whose ball reaches ``release_speed`` moving away from the passer.

    The pass start moves to the first such frame. Turnovers pass through unchanged.
    """
    ok = features[(features["speed"] >= release_speed) & (features["cos"] >= release_cos)]
    release = ok.groupby("pass_idx")["frame"].min()
    is_pass = detected["kind"] == "pass"
    passes = detected[is_pass]
    kept = passes.loc[passes.index.intersection(release.index)].copy()
    kept["start_frame"] = release.loc[kept.index].astype("int64")
    out = pd.concat([kept, detected[~is_pass]])
    return out.sort_values(["match_id", "period", "start_frame"]).reset_index(drop=True)


def detect_passes_v2(
    frames: pd.DataFrame,
    owner: pd.DataFrame,
    min_hold_frames: int,
    max_gap_frames: int | None,
    release_speed: float,
    release_cos: float,
    back_frames: int,
    fwd_frames: int,
) -> pd.DataFrame:
    """v2: transitions gated on the ball accelerating away from the passer.

    ``owner`` should come from ``ball_owner`` with a ``match_radius`` radius;
    ``frames`` need ball ``vx``/``vy``.
    """
    detected = transitions(owner, min_hold_frames, max_gap_frames)
    passes = detected[detected["kind"] == "pass"]
    features = release_features(frames, passes, back_frames, fwd_frames)
    return apply_release_gate(detected, features, release_speed, release_cos)


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
