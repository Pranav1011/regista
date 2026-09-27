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

Also: isotonic and Platt calibrators fitted on games 1-2 at level (a), saved to
eval/calibration_phase1.json (raw pitch control stays the model output; the
isotonic value is the display probability); failed-pass rule sensitivity at
level (a); and the composition of level (b)'s failures, with level (b) scored
with and without tackles and lost dribbles.

Run: uv run python eval/passing_options.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from _common import EVAL_DIR, TEST_GAME, TRAIN_GAMES, load_params, metrica_game
from pass_detection import predict
from scipy.optimize import minimize

from regista.analytics.calibration import fit_isotonic, fit_platt, mean_log_loss
from regista.analytics.kinematics import frame_interval
from regista.analytics.passing_options import (
    build_scene,
    default_params,
    pitch_control_at,
    to_scene_coords,
)
from regista.analytics.possession import release_features
from regista.analytics.shape import goalkeepers
from regista.schema import PITCH_LENGTH_M, PITCH_WIDTH_M

FAILED_PASS_MARKERS = ("INTERCEPTION", "CROSS", "DEEP BALL", "THROUGH BALL", "GOAL KICK")
BLOCK_S = 60.0
BOOTSTRAP_RESAMPLES = 2000
N_BINS = 10
CALIBRATION_PATH = EVAL_DIR / "calibration_phase1.json"


NON_PASS_LOSS = ("THEFT", "FORCED", "HAND BALL", "OFFSIDE", "REFEREE HIT", "END HALF", "WOODWORK")
RULES = {
    "default": "BALL LOST with a pass-attempt subtype (INTERCEPTION, CROSS, DEEP BALL, "
    "THROUGH BALL, GOAL KICK)",
    "interception_only": "BALL LOST with an INTERCEPTION subtype",
    "plus_ball_out": "default, plus BALL LOST (not a non-pass loss) followed directly by BALL OUT "
    "within 3 s or ending outside the lines",
}
BALL_OUT_WINDOW_FRAMES = 75  # 3 s at 25 fps


def labelled_attempts(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Labelled pass attempts with one boolean column per failed-pass rule, plus counts.

    Completed = PASS (offside passes excluded) and belongs to every rule. A failed
    attempt belongs to the rules in ``RULES`` whose definition it meets.
    """
    ev = events.reset_index(drop=True)
    subtype = ev["subtype"].fillna("")
    is_lost = ev["type"] == "BALL LOST"
    nxt = ev.shift(-1)
    to_ball_out = (
        (
            (nxt["type"] == "BALL OUT")
            & ((nxt["start_frame"] - ev["start_frame"]) <= BALL_OUT_WINDOW_FRAMES)
        )
        .fillna(False)
        .to_numpy(bool)
    )
    end_x = ev["end_x"].where(ev["end_x"].notna() | ~to_ball_out, nxt["start_x"])
    end_y = ev["end_y"].where(ev["end_y"].notna() | ~to_ball_out, nxt["start_y"])
    outside = (end_x.abs() > PITCH_LENGTH_M / 2) | (end_y.abs() > PITCH_WIDTH_M / 2)

    completed = (ev["type"] == "PASS") & ~subtype.str.contains("OFFSIDE")
    default = is_lost & subtype.str.contains("|".join(FAILED_PASS_MARKERS))
    interception = is_lost & subtype.str.contains("INTERCEPTION")
    non_pass = subtype.str.contains("|".join(NON_PASS_LOSS))
    ball_out = default | (is_lost & ~non_pass & (to_ball_out | outside.fillna(False)))
    keep = (completed | ball_out) & ev["start_frame"].notna() & ev["start_x"].notna()
    keep &= end_x.notna() & ev["team"].notna()

    out = pd.DataFrame(
        {
            "period": ev["period"].astype(int),
            "frame": ev["start_frame"],
            "team": ev["team"].astype(str),
            "start_x": ev["start_x"],
            "start_y": ev["start_y"],
            "end_x": end_x,
            "end_y": end_y,
            "completed": completed.astype(int),
            "rule_default": completed | default,
            "rule_interception_only": completed | interception,
            "rule_plus_ball_out": completed | ball_out,
        }
    )[keep.to_numpy(bool)]
    out["frame"] = out["frame"].astype(int)
    lost = ev[is_lost]
    counts = {
        "ball_lost_total": len(lost),
        "ball_lost_no_end_location": int(lost["end_x"].isna().sum()),
        "passes_offside_excluded": int(
            ((ev["type"] == "PASS") & subtype.str.contains("OFFSIDE")).sum()
        ),
        **{
            f"failed_{rule}": int((out[f"rule_{rule}"] & (out["completed"] == 0)).sum())
            for rule in RULES
        },
    }
    return out.reset_index(drop=True), counts


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


FAILURE_CATEGORIES = (
    "labelled failed pass",
    "labelled completed pass",
    "tackle or lost dribble",
    "unspecified ball loss",
    "other or unmatched",
)
MATCH_WINDOW_FRAMES = 25  # 1 s at 25 fps


def classify_failures(attempts: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Label each inferred failed attempt by the labelled event nearest its release.

    Uses events within 1 s of the release frame, in priority order: a same-team
    labelled failed pass (default rule), a same-team PASS, a tackle or lost
    dribble (any CHALLENGE, or a same-team THEFT/FORCED ball loss), a same-team
    BALL LOST without subtype, otherwise "other or unmatched". Completed
    attempts get NA. For describing level (b) only; labels never feed the model.
    """
    ev = events.dropna(subset=["start_frame"])
    subtype = ev["subtype"].fillna("")
    kinds = {
        "labelled failed pass": (ev["type"] == "BALL LOST")
        & subtype.str.contains("|".join(FAILED_PASS_MARKERS)),
        "labelled completed pass": ev["type"] == "PASS",
        "tackle or lost dribble": (ev["type"] == "CHALLENGE")
        | ((ev["type"] == "BALL LOST") & subtype.str.contains("THEFT|FORCED")),
        "unspecified ball loss": (ev["type"] == "BALL LOST") & (subtype == ""),
    }
    same_team_only = {"labelled failed pass", "labelled completed pass", "unspecified ball loss"}
    labels = []
    for a in attempts.itertuples():
        if a.completed == 1:
            labels.append(None)
            continue
        near = (ev["period"] == a.period) & (
            (ev["start_frame"] - a.frame).abs() <= MATCH_WINDOW_FRAMES
        )
        label = "other or unmatched"
        for name, mask in kinds.items():
            m = near & mask
            if name in same_team_only:
                m &= ev["team"] == a.team
            if name == "tackle or lost dribble":
                m &= (ev["type"] == "CHALLENGE") | (ev["team"] == a.team)
            if m.any():
                label = name
                break
        labels.append(label)
    return pd.Series(labels, index=attempts.index, dtype="string")


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
    design = np.column_stack([np.ones(len(x)), (x - mean) / std])
    res = minimize(mean_log_loss, np.zeros(design.shape[1]), args=(design, y), jac=True,
                   method="L-BFGS-B")  # fmt: skip
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
                    if name.startswith("pitch_control") and other in ("logistic", base)
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


FEATURES = ["length", "nearest_defender"]


def compare(
    train: pd.DataFrame, test: pd.DataFrame, extra: dict[str, np.ndarray] | None = None
) -> dict:
    """Fit base rate and logistic on ``train``; score pitch control and baselines on ``test``."""
    train = train.dropna(subset=FEATURES)
    test = test.dropna(subset=FEATURES).copy()
    base_rate = float(train["completed"].mean())
    logit = fit_logistic(train[FEATURES].to_numpy(), train["completed"].to_numpy(float))
    test["base_rate"] = base_rate
    test["logistic"] = logit.predict(test[FEATURES].to_numpy())
    preds = {"pitch_control": "pitch_control"}
    for name, fn in (extra or {}).items():
        test[name] = fn(test["pitch_control"].to_numpy())
        preds[name] = name
    preds.update({"logistic": "logistic", "base_rate": "base_rate"})
    y = test["completed"].to_numpy(float)
    return {
        "n_train": len(train),
        "n_test": len(test),
        "completion_rate_train": base_rate,
        "completion_rate_test": float(y.mean()),
        "scores": block_bootstrap(test, preds, base="base_rate"),
        "reliability": {
            name: reliability(test[col].to_numpy(), y)
            for name, col in preds.items()
            if name != "base_rate"
        },
        "logistic_coef": dict(zip(["intercept", *FEATURES], logit.coef.round(3), strict=True)),
        "by_outcome": test.groupby("completed")[["pitch_control", "logistic"]].mean(),
        "test": test,
    }


def evaluate() -> dict:
    params = load_params()
    labelled, inferred, counts = {}, {}, {}
    for game in (*TRAIN_GAMES, TEST_GAME):
        frames, events = metrica_game(game)
        attempts, counts[game] = labelled_attempts(events)
        labelled[game] = score_attempts(frames, attempts)
        b = score_attempts(frames, inferred_attempts(frames, params))
        b["failure_category"] = classify_failures(b, events)
        inferred[game] = b

    def split(data: dict, mask_fn) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.concat([d[mask_fn(d)] for g, d in data.items() if g in TRAIN_GAMES])
        return train.reset_index(drop=True), data[TEST_GAME][mask_fn(data[TEST_GAME])]

    # Level (a): default rule, with calibrators fitted on games 1-2 only
    train_a, test_a = split(labelled, lambda d: d["rule_default"])
    y_train = train_a["completed"].to_numpy(float)
    isotonic = fit_isotonic(train_a["pitch_control"].to_numpy(), y_train)
    platt = fit_platt(train_a["pitch_control"].to_numpy(), y_train)
    CALIBRATION_PATH.write_text(
        json.dumps(
            {
                "fitted_on": list(TRAIN_GAMES),
                "input": "raw pitch control at labelled pass attempts (level a, default rule)",
                "display": "isotonic",
                "isotonic": isotonic.to_dict(),
                "platt": platt.to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    level_a = compare(train_a, test_a, {"pitch_control_isotonic": isotonic.predict,
                                         "pitch_control_platt": platt.predict})  # fmt: skip

    sensitivity = {
        rule: compare(*split(labelled, lambda d, r=rule: d[f"rule_{r}"])) for rule in RULES
    }

    b_all = compare(*split(inferred, lambda d: pd.Series(True, index=d.index)))
    b_no_tackles = compare(
        *split(inferred, lambda d: d["failure_category"].fillna("") != "tackle or lost dribble")
    )
    composition = (
        inferred[TEST_GAME]["failure_category"]
        .fillna("completed")
        .value_counts()
        .rename("attempts")
        .to_frame()
    )
    composition["share"] = composition["attempts"] / composition["attempts"].sum()
    return {
        "level_a": level_a,
        "sensitivity": sensitivity,
        "level_b": {"all": b_all, "without tackles and lost dribbles": b_no_tackles},
        "level_b_composition": composition,
        "counts": counts,
        "calibrators": {"isotonic": isotonic, "platt": platt},
        "labelled": labelled,
        "inferred": inferred,
    }


def _print_compare(title: str, res: dict, reliability_for: tuple[str, ...] = ()) -> None:
    print(f"\n## {title}")
    print(
        f"attempts: train {res['n_train']}, test {res['n_test']}; completion rate "
        f"train {res['completion_rate_train']:.3f}, test {res['completion_rate_test']:.3f}"
    )
    print(res["scores"].round(4).to_markdown())
    for name in reliability_for:
        print(f"reliability, {name}:")
        print(res["reliability"][name].round(3).to_markdown())


def main() -> None:
    r = evaluate()
    for game, c in r["counts"].items():
        print(f"game {game} event counts:", c)
    a = r["level_a"]
    _print_compare(f"Level (a): labelled pass attempts, game {TEST_GAME} held out", a,
                   ("pitch_control", "pitch_control_isotonic", "pitch_control_platt",
                    "logistic"))  # fmt: skip
    print("logistic coefficients (standardised):", a["logistic_coef"])
    print("mean predicted probability by outcome (0 = failed; target = interception point):")
    print(a["by_outcome"].round(3).to_markdown())
    print("\n## Failed-pass rule sensitivity (level a)")
    for rule, res in r["sensitivity"].items():
        sc = res["scores"]
        print(f"- {rule} ({RULES[rule]}): n_test {res['n_test']}, completion "
              f"{res['completion_rate_test']:.3f}; Brier PC {sc.loc['pitch_control', 'brier']:.4f} "
              f"vs logistic {sc.loc['logistic', 'brier']:.4f}; PC - logistic 95% CI "
              f"{sc.loc['pitch_control', 'brier_minus_logistic_ci']}")  # fmt: skip
    print("\n## Level (b) composition, game 3")
    print(r["level_b_composition"].round(3).to_markdown())
    for variant, res in r["level_b"].items():
        _print_compare(f"Level (b), end to end: {variant}", res)


if __name__ == "__main__":
    main()
