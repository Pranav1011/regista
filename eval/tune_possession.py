"""Tune pass-detection parameters on Metrica games 1-2 ONLY. Game 3 is never loaded here.

1. Tolerance k: the smallest k (frames) whose F1, with the best parameters at a
   generous 2 s tolerance, is within 1 point of that plateau.
2. Radius, ball-speed gate, minimum hold, and maximum gap: grid search maximising
   pooled F1 on games 1-2 at that k.

Writes eval/params_phase1.json. Run: uv run python eval/tune_possession.py
"""

from __future__ import annotations

import itertools
import json

import pandas as pd
from _common import PARAMS_PATH, TRAIN_GAMES, metrica_game, true_passes

from regista.analytics.possession import ball_owner, detect_passes, score_passes

RADII_M = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
BALL_SPEED_GATES = (None, 5.0, 7.5, 10.0, 15.0)
MIN_HOLDS = (1, 2, 3, 5)
MAX_GAPS = (None, 25, 50, 125)
TOLERANCES = (1, 2, 3, 5, 8, 12, 18, 25, 50)
PLATEAU_TOL = 50  # 2 s at 25 fps
PLATEAU_SLACK = 0.01


def pooled(scores) -> dict:
    scores = list(scores)
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main() -> None:
    games = {g: metrica_game(g) for g in TRAIN_GAMES}
    truths = {g: true_passes(ev) for g, (_, ev) in games.items()}

    preds: dict[tuple, dict[int, pd.DataFrame]] = {}
    for radius, gate in itertools.product(RADII_M, BALL_SPEED_GATES):
        owners = {g: ball_owner(f, radius, gate) for g, (f, _) in games.items()}
        for hold, gap in itertools.product(MIN_HOLDS, MAX_GAPS):
            preds[(radius, gate, hold, gap)] = {
                g: (p := detect_passes(o, hold, gap))[p["kind"] == "pass"]
                for g, o in owners.items()
            }

    def f1_at(cfg: tuple, k: int) -> dict:
        return pooled(score_passes(preds[cfg][g], truths[g], k) for g in TRAIN_GAMES)

    plateau_cfg = max(preds, key=lambda c: f1_at(c, PLATEAU_TOL)["f1"])
    plateau_f1 = f1_at(plateau_cfg, PLATEAU_TOL)["f1"]
    curve = {k: f1_at(plateau_cfg, k)["f1"] for k in TOLERANCES}
    tol = min(k for k, f1 in curve.items() if f1 >= plateau_f1 - PLATEAU_SLACK)

    rows = [{"cfg": c, **f1_at(c, tol)} for c in preds]
    table = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    radius, gate, hold, gap = table.loc[0, "cfg"]
    best = table.loc[0, ["precision", "recall", "f1"]].astype(float).to_dict()

    params = {
        "tuned_on": list(TRAIN_GAMES),
        "possession": {
            "radius_m": radius,
            "max_ball_speed": gate,
            "min_hold_frames": hold,
            "max_gap_frames": gap,
        },
        "pass_tolerance_frames": tol,
        "tolerance_curve_f1": {str(k): round(v, 4) for k, v in curve.items()},
        "train_scores": {k: round(v, 4) for k, v in best.items()},
        "grid_size": len(preds),
    }
    PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n")
    print(json.dumps(params, indent=2))
    print(table.head(10).to_string())


if __name__ == "__main__":
    main()
