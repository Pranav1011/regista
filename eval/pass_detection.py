"""Score pass detection on the held-out Metrica game 3 and categorise its failures.

Parameters come from eval/params_phase1.json (tuned on games 1-2 only).
Run: uv run python eval/pass_detection.py
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from _common import TEST_GAME, load_params, metrica_game, true_passes

from regista.analytics.possession import ball_owner, detect_passes, score_passes

CONTEXT_FRAMES = 25  # 1 s at 25 fps, used to describe what surrounds a failure


@dataclass
class PassEval:
    game: int
    params: dict
    precision: float
    recall: float
    f1: float
    n_true: int
    n_pred: int
    false_negatives: pd.DataFrame  # true passes missed, with a `reason`
    false_positives: pd.DataFrame  # predicted passes unmatched, with a `reason`
    predicted: pd.DataFrame
    owner: pd.DataFrame


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


def evaluate(game: int = TEST_GAME) -> PassEval:
    params = load_params()
    pp, tol = params["possession"], params["pass_tolerance_frames"]
    frames, events = metrica_game(game)
    truth = true_passes(events)
    owner = ball_owner(frames, pp["radius_m"], pp["max_ball_speed"])
    detected = detect_passes(owner, pp["min_hold_frames"], pp["max_gap_frames"])
    pred = detected[detected["kind"] == "pass"].reset_index(drop=True)
    score = score_passes(pred, truth, tol)

    fn = truth[~score.matched_true.to_numpy()].copy()
    fn["reason"] = [_fn_reason(r, owner, pred, tol) for r in fn.itertuples()]
    fp = pred[~score.matched_pred.to_numpy()].copy()
    fp["reason"] = [_fp_reason(r, events) for r in fp.itertuples()]
    return PassEval(
        game=game,
        params=params,
        precision=score.precision,
        recall=score.recall,
        f1=score.f1,
        n_true=len(truth),
        n_pred=len(pred),
        false_negatives=fn,
        false_positives=fp,
        predicted=pred,
        owner=owner,
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


def main() -> None:
    ev = evaluate()
    print(f"Game {ev.game} (held out): P={ev.precision:.3f} R={ev.recall:.3f} F1={ev.f1:.3f}")
    print(f"true passes={ev.n_true}, predicted passes={ev.n_pred}")
    print("train (games 1-2):", ev.params["train_scores"])
    print(failure_table(ev).to_markdown(index=False))


if __name__ == "__main__":
    main()
