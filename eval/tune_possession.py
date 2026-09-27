"""Tune pass-detection parameters on Metrica games 1-2 ONLY. Game 3 is never loaded here.

v1 (fixed radius; pass starts at the passer's last touch):
1. Tolerance k: the smallest k (frames) whose F1, with the best parameters at a
   generous 2 s tolerance, is within 1 point of that plateau.
2. Radius, ball-speed gate, minimum hold, and maximum gap: grid search maximising
   pooled F1 on games 1-2 at that k.

v2 (designed after the v1 game-3 failure analysis): per-match radius from a
quantile of unlabelled kick-onset distances (ball to nearest player just before
the ball accelerates), plus a release gate (ball speed and direction away from
the passer). The quantile and the remaining thresholds are
grid-searched on games 1-2 at v1's k, so both versions are scored identically.

Writes eval/params_phase1.json.
Run: uv run python eval/tune_possession.py [--version v1|v2|all]
"""

from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd
from _common import PARAMS_PATH, TRAIN_GAMES, metrica_game, true_passes

from regista.analytics.kinematics import frame_interval
from regista.analytics.possession import (
    apply_release_gate,
    ball_owner,
    detect_passes,
    match_radius,
    release_features,
    score_passes,
    transitions,
)

RADII_M = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
BALL_SPEED_GATES = (None, 5.0, 7.5, 10.0, 15.0)
MIN_HOLDS = (1, 2, 3, 5)
MAX_GAPS = (None, 25, 50, 125)
TOLERANCES = (1, 2, 3, 5, 8, 12, 18, 25, 50)
PLATEAU_TOL = 50  # 2 s at 25 fps
PLATEAU_SLACK = 0.01

V2_QUANTILES = (0.8, 0.9, 0.95, 0.98, 0.99)
V2_BALL_SPEED_GATES = (None, 15.0)
V2_MIN_HOLDS = (1, 2, 3)
V2_MAX_GAPS = (None, 50, 125)
V2_RELEASE_SPEEDS = (2.0, 3.0, 4.0, 5.0, 7.0)
V2_RELEASE_COS = (0.0, 0.3, 0.5, 0.7)
V2_RELEASE_BACK_S = 0.2  # fixed by definition, not tuned
V2_RELEASE_FWD_S = 0.4


def pooled(scores) -> dict:
    scores = list(scores)
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _rounded(scores: dict) -> dict:
    return {k: round(float(v), 4) for k, v in scores.items()}


def tune_v1(games: dict, truths: dict) -> dict:
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
    return {
        "possession": {
            "radius_m": radius,
            "max_ball_speed": gate,
            "min_hold_frames": hold,
            "max_gap_frames": gap,
        },
        "pass_tolerance_frames": tol,
        "tolerance_curve_f1": {str(k): round(v, 4) for k, v in curve.items()},
        "train_scores": _rounded(table.loc[0, ["precision", "recall", "f1"]].to_dict()),
        "grid_size": len(preds),
    }


def tune_v2(games: dict, truths: dict, tol: int) -> dict:
    dt = frame_interval(next(iter(games.values()))[0])
    back, fwd = round(V2_RELEASE_BACK_S / dt), round(V2_RELEASE_FWD_S / dt)
    radii = {q: {g: match_radius(f, q) for g, (f, _) in games.items()} for q in V2_QUANTILES}

    results = []
    for q, gate in itertools.product(V2_QUANTILES, V2_BALL_SPEED_GATES):
        owners = {g: ball_owner(games[g][0], radii[q][g], gate) for g in TRAIN_GAMES}
        for hold, gap in itertools.product(V2_MIN_HOLDS, V2_MAX_GAPS):
            detected = {g: transitions(owners[g], hold, gap) for g in TRAIN_GAMES}
            feats = {
                g: release_features(
                    games[g][0], d[d["kind"] == "pass"], back_frames=back, fwd_frames=fwd
                )
                for g, d in detected.items()
            }
            for v, c in itertools.product(V2_RELEASE_SPEEDS, V2_RELEASE_COS):
                scores = []
                for g in TRAIN_GAMES:
                    p = apply_release_gate(detected[g], feats[g], v, c)
                    scores.append(score_passes(p[p["kind"] == "pass"], truths[g], tol))
                results.append({"cfg": (q, gate, hold, gap, v, c), **pooled(scores)})

    table = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
    q, gate, hold, gap, v, c = table.loc[0, "cfg"]
    return {
        "radius_quantile": q,
        "train_radii_m": {str(g): round(radii[q][g], 3) for g in TRAIN_GAMES},
        "max_ball_speed": gate,
        "min_hold_frames": hold,
        "max_gap_frames": gap,
        "release_speed": v,
        "release_cos": c,
        "release_back_s": V2_RELEASE_BACK_S,
        "release_fwd_s": V2_RELEASE_FWD_S,
        "train_scores": _rounded(table.loc[0, ["precision", "recall", "f1"]].to_dict()),
        "grid_size": len(table),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=["v1", "v2", "all"], default="all")
    version = parser.parse_args().version

    games = {g: metrica_game(g) for g in TRAIN_GAMES}
    truths = {g: true_passes(ev) for g, (_, ev) in games.items()}
    params = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}
    params["tuned_on"] = list(TRAIN_GAMES)
    if version in ("v1", "all"):
        v1 = tune_v1(games, truths)
        params["pass_tolerance_frames"] = v1.pop("pass_tolerance_frames")
        params["tolerance_curve_f1"] = v1.pop("tolerance_curve_f1")
        params["v1"] = v1
    if version in ("v2", "all"):
        if "pass_tolerance_frames" not in params:
            raise RuntimeError("v2 is scored at v1's tolerance; run with --version v1 or all first")
        params["v2"] = tune_v2(games, truths, params["pass_tolerance_frames"])
    PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n")
    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
