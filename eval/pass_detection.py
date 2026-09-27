"""Score pass detection on the held-out Metrica game 3 and categorise its failures.

Parameters come from eval/params_phase1.json (tuned on games 1-2 only).
v2 was designed after the v1 game-3 failure analysis; both stay reproducible.
Run: uv run python eval/pass_detection.py [--version v1|v2]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
from _common import TEST_GAME, load_params, metrica_game, true_passes, v2_owner

from regista.analytics.kinematics import frame_interval
from regista.analytics.possession import (
    ball_owner,
    detect_passes,
    detect_passes_v2,
    score_passes,
)

CONTEXT_FRAMES = 25  # 1 s at 25 fps, used to describe what surrounds a failure


@dataclass
class PassEval:
    game: int
    version: str
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


def predict(
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


def evaluate(version: str, game: int = TEST_GAME) -> PassEval:
    params = load_params()
    tol = params["pass_tolerance_frames"]
    frames, events = metrica_game(game)
    truth = true_passes(events)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    ev = evaluate(parser.parse_args().version)
    print(
        f"{ev.version} game {ev.game} (held out): "
        f"P={ev.precision:.3f} R={ev.recall:.3f} F1={ev.f1:.3f} (radius {ev.radius_m:.3f} m)"
    )
    print(f"true passes={ev.n_true}, predicted passes={ev.n_pred}")
    print("train (games 1-2):", ev.params["train_scores"])
    print(failure_table(ev).to_markdown(index=False))
    print(failures_by_team(ev).round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
