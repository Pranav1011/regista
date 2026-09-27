"""Calibration of pitch-control pass success probabilities on held-out Metrica game 3.

Pitch control uses LaurieOnTracking's published parameters; nothing is fitted
for it. Baselines are fitted on games 1-2 only:
- base rate: the pass completion rate on games 1-2 (reported as Brier skill score),
- logistic regression on pass length and nearest-defender distance to the target.

Two levels:
(a) labelled: at every labelled Metrica pass attempt, pitch control at the event's
    end point, from tracking at the start frame. Completed = PASS (offside
    passes excluded); failed = BALL LOST whose subtype marks a pass attempt.
(b) end to end: passes and turnovers inferred by pass detector v2 from tracking
    alone; a turnover counts as a failed pass only if the ball was released
    (accelerated away from the owner, v2's frozen release gate).

Pitch control is evaluated only at target points. Uncertainty: block bootstrap
over 1-minute blocks. Reliability: 10 equal-width bins with counts.

Run: uv run python eval/passing_options.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from _common import TEST_GAME, TRAIN_GAMES, load_params, metrica_game
from pass_detection import predict
from scipy.optimize import minimize

from regista.analytics.kinematics import frame_interval
from regista.analytics.passing_options import (
    build_scene,
    default_params,
    pitch_control_at,
    to_scene_coords,
)
from regista.analytics.possession import release_features
from regista.analytics.shape import goalkeepers

FAILED_PASS_MARKERS = ("INTERCEPTION", "CROSS", "DEEP BALL", "THROUGH BALL", "GOAL KICK")
BLOCK_S = 60.0
BOOTSTRAP_RESAMPLES = 2000
N_BINS = 10
LEVELS = ("labelled", "end_to_end")


def labelled_attempts(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pass attempts from events: completed PASS and pass-like BALL LOST. Also exclusion counts."""
    ev = events.dropna(subset=["start_frame", "start_x", "end_x", "team"])
    subtype = ev["subtype"].fillna("")
    completed = (ev["type"] == "PASS") & ~subtype.str.contains("OFFSIDE")
    failed = (ev["type"] == "BALL LOST") & subtype.str.contains("|".join(FAILED_PASS_MARKERS))
    lost = events[events["type"] == "BALL LOST"]
    excluded = {
        "ball_lost_total": len(lost),
        "ball_lost_used_as_failed_pass": int(failed.sum()),
        "ball_lost_no_end_location": int(lost["end_x"].isna().sum()),
        "passes_offside_excluded": int(
            (
                (events["type"] == "PASS") & events["subtype"].fillna("").str.contains("OFFSIDE")
            ).sum()
        ),
    }
    att = ev[completed | failed]
    out = pd.DataFrame(
        {
            "period": att["period"].astype(int),
            "frame": att["start_frame"].astype(int),
            "team": att["team"].astype(str),
            "start_x": att["start_x"],
            "start_y": att["start_y"],
            "end_x": att["end_x"],
            "end_y": att["end_y"],
            "completed": completed[completed | failed].astype(int),
        }
    )
    return out.reset_index(drop=True), excluded


def inferred_attempts(frames: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Pass attempts from v2 detections: passes, plus turnovers with a ball release."""
    owner, detected, _ = predict(frames, "v2", params)
    pp = params["v2"]
    dt = frame_interval(frames)
    turnovers = detected[detected["kind"] == "turnover"]
    feats = release_features(frames, turnovers, round(pp["release_back_s"] / dt),
                             round(pp["release_fwd_s"] / dt))  # fmt: skip
    ok = feats[(feats["speed"] >= pp["release_speed"]) & (feats["cos"] >= pp["release_cos"])]
    release = ok.groupby("pass_idx")["frame"].min()
    failed = turnovers.loc[turnovers.index.intersection(release.index)].copy()
    failed["start_frame"] = release.loc[failed.index].to_numpy()
    attempts = pd.concat([detected[detected["kind"] == "pass"], failed])

    ball = frames[frames["team"] == "ball"].set_index(["period", "frame"])[["x", "y"]]
    start = ball.reindex(pd.MultiIndex.from_arrays([attempts["period"], attempts["start_frame"]]))
    end = ball.reindex(pd.MultiIndex.from_arrays([attempts["period"], attempts["end_frame"]]))
    out = pd.DataFrame(
        {
            "period": attempts["period"].astype(int).to_numpy(),
            "frame": attempts["start_frame"].astype(int).to_numpy(),
            "team": attempts["team"].astype(str).to_numpy(),
            "start_x": start["x"].to_numpy(),
            "start_y": start["y"].to_numpy(),
            "end_x": end["x"].to_numpy(),
            "end_y": end["y"].to_numpy(),
            "completed": (attempts["kind"] == "pass").astype(int).to_numpy(),
        }
    )
    return out.dropna(subset=["start_x", "end_x"]).reset_index(drop=True)


def score_attempts(frames: pd.DataFrame, attempts: pd.DataFrame) -> pd.DataFrame:
    """Add pitch control at the target, pass length, nearest-defender distance, and time."""
    p = default_params()
    gk = goalkeepers(frames).set_index(["period", "team"])["gk_id"]
    indexed = frames.set_index(["period", "frame"]).sort_index()
    times = indexed["t"].groupby(level=[0, 1]).first()
    pcs, lengths, nearest = [], [], []
    for a in attempts.itertuples():
        rows = indexed.loc[(a.period, a.frame)].reset_index()
        gks = {team: gk.get((a.period, team)) for team in ("home", "away")}
        scene = build_scene(rows, a.period, a.frame, a.team, gks,
                            ball_xy=np.array([a.start_x, a.start_y]), p=p)  # fmt: skip
        target = to_scene_coords(np.array([a.end_x, a.end_y]), a.team)
        pcs.append(pitch_control_at(target, scene, p))
        lengths.append(float(np.hypot(a.end_x - a.start_x, a.end_y - a.start_y)))
        d = np.linalg.norm(scene.def_pos - target, axis=1)
        nearest.append(float(d.min()) if len(d) else np.nan)
    out = attempts.copy()
    out["pitch_control"] = pcs
    out["length"] = lengths
    out["nearest_defender"] = nearest
    out["t"] = times.reindex(pd.MultiIndex.from_arrays([out["period"], out["frame"]])).to_numpy()
    return out


@dataclass
class Logistic:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray  # intercept first

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(self.coef[0] + z @ self.coef[1:])))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> Logistic:
    """Unregularised logistic regression by maximum likelihood (standardised features)."""
    mean, std = x.mean(axis=0), x.std(axis=0)
    z = (x - mean) / std

    def nll(w: np.ndarray) -> float:
        logits = w[0] + z @ w[1:]
        return float(np.sum(np.logaddexp(0.0, logits) - y * logits))

    res = minimize(nll, np.zeros(z.shape[1] + 1), method="BFGS")
    if not res.success:
        raise RuntimeError(f"logistic regression did not converge: {res.message}")
    return Logistic(mean, std, res.x)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def block_bootstrap(
    df: pd.DataFrame, preds: dict[str, str], base: str, seed: int = 0
) -> pd.DataFrame:
    """Brier (and skill vs ``base``) with 95% CIs, resampling 1-minute blocks."""
    blocks = df["period"].astype(str) + ":" + (df["t"] // BLOCK_S).astype(int).astype(str)
    codes, uniq = pd.factorize(blocks)
    y = df["completed"].to_numpy(float)
    sq = {name: (df[col].to_numpy(float) - y) ** 2 for name, col in preds.items()}
    sums = {name: np.bincount(codes, weights=v, minlength=len(uniq)) for name, v in sq.items()}
    counts = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(BOOTSTRAP_RESAMPLES, len(uniq)))
    n = counts[idx].sum(axis=1)
    boot = {name: s[idx].sum(axis=1) / n for name, s in sums.items()}
    rows = []
    for name in preds:
        point = sq[name].mean()
        diff_vs = {other: boot[name] - boot[other] for other in preds if other != name}
        skill = 1 - boot[name] / boot[base]
        rows.append(
            {
                "model": name,
                "brier": point,
                "brier_ci_low": np.percentile(boot[name], 2.5),
                "brier_ci_high": np.percentile(boot[name], 97.5),
                "skill_vs_base_rate": 1 - point / sq[base].mean(),
                "skill_ci_low": np.percentile(skill, 2.5),
                "skill_ci_high": np.percentile(skill, 97.5),
                **{
                    f"brier_minus_{other}_ci": (
                        f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
                    )
                    for other, d in diff_vs.items()
                    if name == "pitch_control"
                },
            }
        )
    return pd.DataFrame(rows).set_index("model")


def reliability(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Equal-width bins: count, mean predicted probability, observed completion rate."""
    bins = np.clip((p * N_BINS).astype(int), 0, N_BINS - 1)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y})
    g = df.groupby("bin")
    table = pd.DataFrame({"count": g.size(), "mean_predicted": g["p"].mean(),
                          "observed": g["y"].mean()})  # fmt: skip
    table.index = [f"{b / N_BINS:.1f}-{(b + 1) / N_BINS:.1f}" for b in table.index]
    return table


def evaluate() -> dict:
    params = load_params()
    data: dict[tuple[str, int], pd.DataFrame] = {}
    exclusions = {}
    for game in (*TRAIN_GAMES, TEST_GAME):
        frames, events = metrica_game(game)
        labelled, exclusions[game] = labelled_attempts(events)
        data[("labelled", game)] = score_attempts(frames, labelled)
        data[("end_to_end", game)] = score_attempts(frames, inferred_attempts(frames, params))

    results = {}
    for level in LEVELS:
        train = pd.concat([data[(level, g)] for g in TRAIN_GAMES], ignore_index=True)
        test = data[(level, TEST_GAME)].copy()
        features = ["length", "nearest_defender"]
        train = train.dropna(subset=features)
        test = test.dropna(subset=features)
        base_rate = float(train["completed"].mean())
        logit = fit_logistic(train[features].to_numpy(), train["completed"].to_numpy(float))
        test["base_rate"] = base_rate
        test["logistic"] = logit.predict(test[features].to_numpy())
        y = test["completed"].to_numpy(float)
        results[level] = {
            "n_train": len(train),
            "n_test": len(test),
            "completion_rate_train": base_rate,
            "completion_rate_test": float(y.mean()),
            "scores": block_bootstrap(
                test,
                {
                    "pitch_control": "pitch_control",
                    "logistic": "logistic",
                    "base_rate": "base_rate",
                },
                base="base_rate",
            ),
            "reliability_pitch_control": reliability(test["pitch_control"].to_numpy(), y),
            "reliability_logistic": reliability(test["logistic"].to_numpy(), y),
            "logistic_coef": dict(zip(["intercept", *features], logit.coef.round(3), strict=True)),
            "test": test,
        }
    return {"levels": results, "exclusions": exclusions, "data": data}


def main() -> None:
    r = evaluate()
    print("Failed-pass rule: BALL LOST with subtype containing", FAILED_PASS_MARKERS)
    for game, ex in r["exclusions"].items():
        print(f"game {game} event exclusions:", ex)
    for level, res in r["levels"].items():
        print(f"\n## Level: {level} (game {TEST_GAME} held out)")
        print(
            f"attempts: train {res['n_train']}, test {res['n_test']}; completion rate "
            f"train {res['completion_rate_train']:.3f}, test {res['completion_rate_test']:.3f}"
        )
        print("logistic coefficients (standardised):", res["logistic_coef"])
        print(res["scores"].round(4).to_markdown())
        by_outcome = res["test"].groupby("completed")[["pitch_control", "logistic"]].mean()
        print("mean predicted probability by outcome (0 = failed, target = interception point):")
        print(by_outcome.round(3).to_markdown())
        print("reliability, pitch control:")
        print(res["reliability_pitch_control"].round(3).to_markdown())
        print("reliability, logistic:")
        print(res["reliability_logistic"].round(3).to_markdown())


if __name__ == "__main__":
    main()
