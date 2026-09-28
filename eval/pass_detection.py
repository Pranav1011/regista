"""Score pass detection on the held-out Metrica game 3 and categorise its failures.

Parameters come from eval/params_phase1.json (tuned on games 1-2 only).
v2 was designed after the v1 game-3 failure analysis; both stay reproducible.
Run: uv run python eval/pass_detection.py [--version v1|v2]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from _common import TEST_GAME, load_params, metrica_game, true_passes

from regista.analytics.possession import score_passes
from regista.pipeline import detect

CONTEXT_FRAMES = 25  # 1 s at 25 fps, used to describe what surrounds a failure


@dataclass
class PassEval:
    game: int
    version: str
    labels: str  # "corrected" or "raw" event player ids
    params: dict  # the version's parameters
    radius_m: float
    precision: float
    recall: float
    f1: float
    n_true: int
    n_pred: int
    false_negatives: pd.DataFrame  # true passes missed, with a `reason`
    false_positives: pd.DataFrame  # predicted passes unmatched, with a `reason`
    predicted: pd.DataFrame
    owner: pd.DataFrame
    truth: pd.DataFrame  # labelled passes scored against


def _owned(owner: pd.DataFrame, period: int, player: str, lo: int, hi: int) -> bool:
    w = owner[(owner["period"] == period) & owner["frame"].between(lo, hi)]
    return bool((w["owner_id"] == player).any())


def _fn_reason(row, owner: pd.DataFrame, pred: pd.DataFrame, tol: int) -> str:
    s, e, per = int(row.start_frame), int(row.end_frame), row.period
    if not _owned(owner, per, row.to_player, s, max(e, s) + tol):
        return "receiver never within radius (one-touch lay-off, deflection or loose receipt)"
    if not _owned(owner, per, row.from_player, s - CONTEXT_FRAMES, s + tol):
        return "passer never within radius before release"
    near = pred[(pred["period"] == per) & (pred["start_frame"] - s).abs().le(CONTEXT_FRAMES)]
    if ((near["from_player"] == row.from_player) & (near["to_player"] == row.to_player)).any():
        return "right players, release time off by more than the tolerance"
    if (near["from_player"] == row.from_player).any() or (near["to_player"] == row.to_player).any():
        return "wrong passer or receiver (a third player came within radius)"
    return "possession chain broken (ball near an opponent or a brief touch in between)"


def _fp_reason(row, events: pd.DataFrame) -> str:
    near = events[
        (events["period"] == row.period)
        & (events["start_frame"] - row.start_frame).abs().le(CONTEXT_FRAMES)
        & (events["type"] != "PASS")
    ]
    if near.empty:
        return "no labelled event within 1 s (dribble near a teammate or unlabelled touch)"
    return f"near a labelled {near.iloc[0]['type']} event, not a PASS"


predict = detect  # the frozen pipeline lives in regista.pipeline


def evaluate(version: str, game: int = TEST_GAME, labels: str = "corrected") -> PassEval:
    params = load_params()
    tol = params["pass_tolerance_frames"]
    frames, events = metrica_game(game)
    truth = true_passes(events, labels)
    owner, detected, radius = predict(frames, version, params)
    pred = detected[detected["kind"] == "pass"].reset_index(drop=True)
    score = score_passes(pred, truth, tol)

    fn = truth[~score.matched_true.to_numpy()].copy()
    fn["reason"] = [_fn_reason(r, owner, pred, tol) for r in fn.itertuples()]
    fp = pred[~score.matched_pred.to_numpy()].copy()
    fp["reason"] = [_fp_reason(r, events) for r in fp.itertuples()]
    return PassEval(
        game=game,
        version=version,
        labels=labels,
        params=params[version],
        radius_m=radius,
        precision=score.precision,
        recall=score.recall,
        f1=score.f1,
        n_true=len(truth),
        n_pred=len(pred),
        false_negatives=fn,
        false_positives=fp,
        predicted=pred,
        owner=owner,
        truth=truth,
    )


def failure_table(ev: PassEval, n_examples: int = 3) -> pd.DataFrame:
    """Failure modes ranked by count, with example frame ranges."""
    rows = []
    for kind, df in (("missed (FN)", ev.false_negatives), ("spurious (FP)", ev.false_positives)):
        for reason, g in df.groupby("reason"):
            examples = ", ".join(
                f"P{r.period} {int(r.start_frame)}-{int(r.end_frame)}"
                for r in g.head(n_examples).itertuples()
            )
            rows.append({"type": kind, "reason": reason, "count": len(g), "examples": examples})
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def failures_by_team(ev: PassEval) -> pd.DataFrame:
    """Missed passes per team and reason, as counts and as a share of that team's true passes."""
    true_per_team = ev.truth.groupby("team").size().rename("true_passes")
    table = ev.false_negatives.groupby(["team", "reason"]).size().rename("missed").reset_index()
    table = table.join(true_per_team, on="team")
    table["share_of_team_passes"] = table["missed"] / table["true_passes"]
    return table.sort_values(["reason", "team"]).reset_index(drop=True)


def release_distance_share(game: int, labels: str, threshold_m: float = 0.5) -> float:
    """Share of labelled pass releases with the ball more than ``threshold_m`` from the passer."""
    frames, events = metrica_game(game)
    truth = true_passes(events, labels)
    idx = frames.set_index(["period", "frame", "player_id"])[["x", "y"]]
    frame_ids = truth["start_frame"].astype(int)
    passer = idx.reindex(pd.MultiIndex.from_arrays([truth["period"], frame_ids,
                                                    truth["from_player"]])).to_numpy()  # fmt: skip
    ball = idx.reindex(pd.MultiIndex.from_arrays([truth["period"], frame_ids,
                                                  ["ball"] * len(truth)])).to_numpy()  # fmt: skip
    dist = np.hypot(*(passer - ball).T)
    dist = dist[~np.isnan(dist)]
    return float((dist > threshold_m).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    version = parser.parse_args().version
    raw = evaluate(version, labels="raw")
    print(
        f"{version} game {raw.game} (held out), published labels: "
        f"P={raw.precision:.3f} R={raw.recall:.3f} F1={raw.f1:.3f}"
    )
    ev = evaluate(version, labels="corrected")
    print(
        f"{ev.version} game {ev.game} (held out), corrected labels: "
        f"P={ev.precision:.3f} R={ev.recall:.3f} F1={ev.f1:.3f} (radius {ev.radius_m:.3f} m)"
    )
    print(f"true passes={ev.n_true}, predicted passes={ev.n_pred}")
    for game in (1, 2, 3):
        shares = {lab: release_distance_share(game, lab) for lab in ("raw", "corrected")}
        print(f"game {game}: releases with ball > 0.5 m from passer:",
              {k: round(v, 3) for k, v in shares.items()})  # fmt: skip
    print("train (games 1-2):", ev.params["train_scores"])
    print(failure_table(ev).to_markdown(index=False))
    print(failures_by_team(ev).round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
