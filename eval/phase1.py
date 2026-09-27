"""Regenerate the Phase 1 report end to end: eval/reports/phase1.md and docs/img/phase1/*.png.

Every number in the report is computed here (or by the eval modules this calls);
none is typed by hand. Tuning is re-run and must reproduce eval/params_phase1.json
exactly, otherwise this script raises.

Run: uv run python eval/phase1.py [--skip-tuning-check]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plots
from _common import (
    PARAMS_PATH,
    REPORTS_DIR,
    SKILLCORNER_MATCHES,
    TEST_GAME,
    TRAIN_GAMES,
    event_id_mismatches,
    load_params,
    metrica_game,
    possession_phase,
    true_passes,
)
from formations_eval import (
    BOOTSTRAP_RESAMPLES,
    FULL_MATCH_PRESENCE,
    metrica_diagnostics,
    skillcorner_roles,
)
from pass_detection import evaluate as evaluate_passes
from pass_detection import failure_table, release_distance_share
from passing_options import FAILED_PASS_MARKERS, RULES
from passing_options import evaluate as evaluate_options
from shape_networks import edge_residuals
from shape_networks import evaluate as evaluate_shape_networks

from regista import io
from regista.analytics.formations import detect_formations, window_shapes
from regista.analytics.pass_network import pass_edges
from regista.analytics.passing_options import (
    build_scene,
    passing_options,
    pitch_control_grid,
)
from regista.analytics.shape import goalkeepers
from regista.schema import Source

REPO = Path(__file__).resolve().parent.parent
IMG_DIR = REPO / "docs" / "img" / "phase1"
REPORT_PATH = REPORTS_DIR / "phase1.md"


def f(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def ci(lo: float, hi: float, digits: int = 3, signed: bool = False) -> str:
    fmt = f"+.{digits}f" if signed else f".{digits}f"
    return f"{lo:{fmt}} to {hi:{fmt}}"


def md(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    return df.round(digits).to_markdown(index=index)


def _ci_upper(text: str) -> float:
    """Upper bound from a '[lo, hi]' interval string."""
    return float(text.strip("[]").split(",")[1])


def verify_tuning() -> None:
    """Re-run tuning on games 1-2 and require it to match the committed parameters."""
    from tune_possession import tune_v1, tune_v2

    committed = load_params()
    games = {g: metrica_game(g) for g in TRAIN_GAMES}
    truths = {g: true_passes(ev) for g, (_, ev) in games.items()}
    v1 = tune_v1(games, truths)
    tol = v1.pop("pass_tolerance_frames")
    v1.pop("tolerance_curve_f1")
    v2 = tune_v2(games, truths, tol)
    fresh = json.loads(json.dumps({"tol": tol, "v1": v1, "v2": v2}))
    stored = {"tol": committed["pass_tolerance_frames"], "v1": committed["v1"],
              "v2": committed["v2"]}  # fmt: skip
    if fresh != stored:
        raise RuntimeError(f"re-tuning does not reproduce {PARAMS_PATH}:\n{fresh}\nvs\n{stored}")


def ingest_summary() -> pd.DataFrame:
    rows = []
    for source, ids in ((Source.METRICA, [str(g) for g in (*TRAIN_GAMES, TEST_GAME)]),
                        (Source.SKILLCORNER, SKILLCORNER_MATCHES)):  # fmt: skip
        for m in ids:
            info = io.read_info(source, m)
            d = info["rows_dropped_out_of_bounds"]
            rows.append(
                {
                    "source": source.value,
                    "match": m,
                    "fps": info["frame_rate"],
                    "rows": info["rows"],
                    "oob_ball_rows_dropped": d["ball"],
                    "oob_player_rows_dropped": d["player"],
                    "flipped_periods": ",".join(map(str, info["flipped_periods"])) or "-",
                }
            )
    return pd.DataFrame(rows)


def formation_windows(params: dict) -> list[dict]:
    """Four game-3 snapshots (home/away x in/out of possession) from the 5-10 min window."""
    frames, _ = metrica_game(TEST_GAME)
    shapes = window_shapes(frames, possession_phase(frames, params), 300.0, 300.0)
    formations, roles = detect_formations(shapes)
    out = []
    for team in ("home", "away"):
        for phase in ("out", "in"):
            key = {"match_id": str(TEST_GAME), "period": 1, "team": team, "phase": phase,
                   "window": 1}  # fmt: skip
            sel = np.logical_and.reduce([shapes[k] == v for k, v in key.items()])
            fsel = np.logical_and.reduce([formations[k] == v for k, v in key.items()])
            rsel = np.logical_and.reduce([roles[k] == v for k, v in key.items()])
            s, fr = shapes[sel], formations[fsel].iloc[0]
            r = roles[rsel].set_index("player_id").loc[s["player_id"]]
            out.append(
                {
                    "name": f"game {TEST_GAME} {team}, {phase} of possession, 5-10 min",
                    "xy": s[["x", "y"]].to_numpy(),
                    "roles": r["role"].tolist(),
                    "label": fr["label"],
                    "runner_up": fr["runner_up"],
                    "margin": fr["margin"],
                }
            )
    return out


def options_example(level_a_test: pd.DataFrame) -> tuple:
    """The game-3 completed pass whose raw pitch control is closest to 0.5 in period 1."""
    frames, _ = metrica_game(TEST_GAME)
    cand = level_a_test[(level_a_test["completed"] == 1) & (level_a_test["period"] == 1)]
    a = cand.iloc[(cand["pitch_control"] - 0.5).abs().argsort().iloc[0]]
    gk = goalkeepers(frames).set_index(["period", "team"])["gk_id"]
    gks = {t: gk.get((int(a["period"]), t)) for t in ("home", "away")}
    rows = frames[(frames["period"] == a["period"]) & (frames["frame"] == a["frame"])]
    scene = build_scene(rows, int(a["period"]), int(a["frame"]), a["team"], gks,
                        ball_xy=np.array([a["start_x"], a["start_y"]]))  # fmt: skip
    surface, xg, yg = pitch_control_grid(scene)
    title = (f"Passing options, game {TEST_GAME} period {int(a['period'])} frame {int(a['frame'])}"
             f" ({a['team']} attacking left to right)")  # fmt: skip
    return surface, xg, yg, scene, passing_options(scene), title


def build_report(skip_tuning_check: bool) -> str:
    params = load_params()
    if not skip_tuning_check:
        verify_tuning()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- data integrity -------------------------------------------------------------
    ingest = ingest_summary()
    swaps = {g: event_id_mismatches(*metrica_game(g), "from_player_raw")
             for g in (*TRAIN_GAMES, TEST_GAME)}  # fmt: skip
    swaps_after = {g: len(event_id_mismatches(*metrica_game(g))) for g in swaps}
    swap3 = swaps[TEST_GAME]
    swap_events = int(swap3["events"].sum())

    # ---- pass detection -------------------------------------------------------------
    pe = {(v, lab): evaluate_passes(v, labels=lab) for v in ("v1", "v2")
          for lab in ("raw", "corrected")}  # fmt: skip
    pass_table = pd.DataFrame(
        [
            {
                "detector": v,
                "labels": {"raw": "published", "corrected": "corrected"}[lab],
                "precision": e.precision,
                "recall": e.recall,
                "f1": e.f1,
                "radius_m": e.radius_m,
                "predicted": e.n_pred,
                "labelled": e.n_true,
            }
            for (v, lab), e in pe.items()
        ]
    )
    raw_v1 = pe[("v1", "raw")]
    swap_pair = set(swap3["credited"])
    fn_swap = int(
        ((raw_v1.false_negatives["period"] == 2)
         & (raw_v1.false_negatives["from_player"].isin(swap_pair)
            | raw_v1.false_negatives["to_player"].isin(swap_pair))).sum()
    )  # fmt: skip
    release = pd.DataFrame(
        {
            lab: {g: release_distance_share(g, lab) for g in (*TRAIN_GAMES, TEST_GAME)}
            for lab in ("raw", "corrected")
        }  # fmt: skip
    ).rename(columns={"raw": "published labels", "corrected": "corrected labels"})
    release.index.name = "game"
    failures = failure_table(pe[("v2", "corrected")])

    # ---- shape and networks ---------------------------------------------------------
    sn = evaluate_shape_networks()
    sim = sn["network_similarity"]
    sim = sim.assign(labels=sim["labels"].replace({"raw": "published"}))
    sim3 = sim[sim["game"] == TEST_GAME][["labels", "version", "team", "edges_a", "edges_b",
                                          "pearson", "spearman", "weighted_jaccard"]]  # fmt: skip
    home_v1 = sim3[(sim3["team"] == "home") & (sim3["version"] == "v1")].set_index("labels")
    home_v1 = home_v1.rename(index={"published": "raw"})
    frames3, events3 = metrica_game(TEST_GAME)
    nodes3 = sn["networks"][(TEST_GAME, "v2")][0]
    raw_edges = pass_edges(true_passes(events3, "raw").assign(match_id=str(TEST_GAME)))
    plots.pass_networks(
        {
            "inferred from tracking (v2)": sn["networks"][(TEST_GAME, "v2")],
            "events, published labels": (nodes3, raw_edges),
            "events, corrected labels": sn["networks"][(TEST_GAME, "events")],
        },
        "home",
        f"Game {TEST_GAME} home pass network",
        IMG_DIR / "pass_networks_game3_home.png",
    )
    residuals = edge_residuals(sn["networks"], TEST_GAME, "v2", "home")

    # ---- formations -----------------------------------------------------------------
    md_ = metrica_diagnostics(params)
    sc = skillcorner_roles(params)
    agree = sc["agreement"]
    boot = sc["bootstrap"]
    plots.formation_snapshots(formation_windows(params), IMG_DIR / "formations_game3.png")
    stab = sc["stability"].groupby("phase")["stability"].median()
    back = sc["back_line_stability"].groupby("phase")["stability"].median()
    side_full = sc["side_by_period_full_match"]
    side_all = (sc["roles"].dropna(subset=["side_template_ok"]).groupby("period")
                ["side_template_ok"].agg(lambda s: s.astype(bool).mean()))  # fmt: skip

    # ---- passing options ------------------------------------------------------------
    po = evaluate_options()
    a = po["level_a"]
    a_sc = a["scores"]
    plots.reliability_diagram(
        {
            "raw pitch control": a["reliability"]["pitch_control"],
            "isotonic (fit on games 1-2)": a["reliability"]["pitch_control_isotonic"],
            "Platt (fit on games 1-2)": a["reliability"]["pitch_control_platt"],
            "logistic baseline": a["reliability"]["logistic"],
        },
        f"Reliability, labelled pass attempts, game {TEST_GAME} (held out)",
        IMG_DIR / "reliability_level_a.png",
    )
    plots.passing_options_snapshot(*options_example(a["test"]),
                                   IMG_DIR / "passing_options_example.png")  # fmt: skip
    b_all = po["level_b"]["all"]
    b_clean = po["level_b"]["without tackles and lost dribbles"]
    comp = po["level_b_composition"]
    iso_zero = a["test"].assign(iso=po["calibrators"]["isotonic"].predict(
        a["test"]["pitch_control"].to_numpy()))  # fmt: skip
    iso_zero = iso_zero[iso_zero["iso"] <= 1e-9]

    # ---- TL;DR --------------------------------------------------------------------
    v1r, v1c = pe[("v1", "raw")], pe[("v1", "corrected")]
    v2r, v2c = pe[("v2", "raw")], pe[("v2", "corrected")]
    gs = boot.loc["group strict: template - depth"]
    mid, side = boot.loc["listed MID strict: template - depth"], boot.loc["side: template - thirds"]
    pc, lg = a_sc.loc["pitch_control"], a_sc.loc["logistic"]
    beats = {
        rule: _ci_upper(res["scores"].loc["pitch_control", "brier_minus_logistic_ci"]) < 0
        for rule, res in po["sensitivity"].items()
    }
    rule_sentence = (
        f"Pitch control beats the baseline under all {len(beats)} failed-pass rules tested."
        if all(beats.values())
        else "Pitch control does not beat the baseline under: "
        + ", ".join(r for r, ok in beats.items() if not ok)
        + "."
    )
    tuning_sentence = (
        "Tuning data (games 1-2) needed no correction, so no parameter depended on it."
        if sum(len(swaps[g]) for g in TRAIN_GAMES) == 0
        else "WARNING: tuning data contains id mismatches."
    )
    stability_table = pd.DataFrame({"exact label": stab, "back-line count": back}).T
    narrow_share = md_["away_352_narrow"] / md_["away_352_windows"]
    narrow_sentence = (
        "The margin is narrow in a minority of windows, not in most."
        if narrow_share < 0.5
        else "The margin is narrow in most windows."
    )
    iso, platt = a_sc.loc["pitch_control_isotonic"], a_sc.loc["pitch_control_platt"]
    by_out = a["by_outcome"]
    tldr = f"""## TL;DR

- **Label error found and corrected.** Metrica game {TEST_GAME}'s event file swaps players
  {" and ".join(sorted(swap_pair))} in the second half: all {swap_events} of their events start
  on the other player's tracked position (median {f(swap3["median_dist_to_nearest"].max(), 1)} m).
  No other player-half in games 1-3 is mismatched. With the published labels, pass
  detection on held-out game {TEST_GAME} scores F1 {f(v1r.f1)} (v1) / {f(v2r.f1)} (v2); with
  corrected labels, {f(v1c.f1)} / {f(v2c.f1)}, in line with training
  ({f(params["v1"]["train_scores"]["f1"])} / {f(params["v2"]["train_scores"]["f1"])}). The swap
  alone caused {fn_swap} of v1's {len(v1r.false_negatives)} misses. Both label sets are reported.
- **Formation templates about tie a depth-rank baseline on position groups.** Against
  SkillCorner's listed positions ({len(SKILLCORNER_MATCHES)} matches), strict group agreement is
  {f(agree.loc["all", "group_template_strict"])} vs {f(agree.loc["all", "group_depth_strict"])}
  (difference {gs["difference"]:+.3f}, 95% CI {ci(gs["ci_low"], gs["ci_high"], signed=True)}).
  Templates add value on sides ({f(agree.loc["all", "side_template"])} vs
  {f(agree.loc["all", "side_thirds"])} for width thirds, {side["difference"]:+.3f}, CI
  {ci(side["ci_low"], side["ci_high"], signed=True)}) and on midfielders ({mid["difference"]:+.3f},
  CI {ci(mid["ci_low"], mid["ci_high"], signed=True)}). Exact template labels are noisy at
  5 minutes (label stability median {f(stab.median(), 2)}); role groups and sides are the
  trusted output.
- **Pitch control beats a logistic baseline; post-hoc calibration improves it further.** At
  labelled pass attempts in game {TEST_GAME} (n = {a["n_test"]}), raw pitch control has Brier
  {f(pc["brier"], 4)} (skill vs base rate {f(pc["skill_vs_base_rate"])}) against
  {f(lg["brier"], 4)} ({f(lg["skill_vs_base_rate"])}) for logistic regression on pass length and
  nearest-defender distance; the paired difference is {pc["brier_minus_logistic_ci"]}.
  Calibrators fitted on games 1-2 lower Brier to {f(iso["brier"], 4)} (isotonic) and
  {f(platt["brier"], 4)} (Platt). {rule_sentence}
- **Known limitations.** Failed-pass targets are interception points, which pushes pitch
  control low for failures (mean {f(by_out.loc[0, "pitch_control"])} vs
  {f(by_out.loc[1, "pitch_control"])} for completed passes); options for teammates who were
  not passed to cannot be validated; one held-out game; Metrica has no formation labels; v2
  was designed after the game-{TEST_GAME} failure analysis, which turned out to be mostly the
  label swap. Full list below.
"""

    # ---- report body ----------------------------------------------------------------
    sens = pd.DataFrame(
        [
            {
                "rule": rule,
                "definition": RULES[rule],
                "failed (train)": int(sum(po["counts"][g][f"failed_{rule}"] for g in TRAIN_GAMES)),
                "failed (test)": po["counts"][TEST_GAME][f"failed_{rule}"],
                "Brier pitch control": res["scores"].loc["pitch_control", "brier"],
                "Brier logistic": res["scores"].loc["logistic", "brier"],
                "PC - logistic, 95% CI": res["scores"].loc[
                    "pitch_control", "brier_minus_logistic_ci"
                ],
            }
            for rule, res in po["sensitivity"].items()
        ]
    )
    ball_out_added = {g: po["counts"][g]["failed_plus_ball_out"] - po["counts"][g]["failed_default"]
                      for g in po["counts"]}  # fmt: skip

    data_rows = "\n".join(
        [
            "| Metrica Sports sample games 1-3 (tracking + events) | Pass detection (tune on"
            " 1-2, test on 3), formations, shape, networks, pitch control |"
            " Acknowledge Metrica Sports |",
            f"| SkillCorner open data, {len(SKILLCORNER_MATCHES)} matches (broadcast tracking) |"
            " Role validation against listed positions | MIT, credit SkillCorner |",
        ]
    )
    oob_cols = ["rows", "oob_ball_rows_dropped", "oob_player_rows_dropped"]
    oob_totals = ingest.groupby("source")[oob_cols].sum()
    train_games = ", ".join(str(g) for g in TRAIN_GAMES)
    train_mismatches = sum(len(swaps[g]) for g in TRAIN_GAMES)
    residual_cols = ["from_player", "to_player", "passes_inferred", "passes_events", "residual"]
    residual_table = residuals.head(6)[residual_cols]
    metrica_stability = md_["stability"].merge(
        md_["back_line_stability"], on=["match_id", "team", "phase"], suffixes=("", "_back_line")
    )[["match_id", "team", "phase", "stability", "stability_back_line"]]

    def score_table(res: dict) -> pd.DataFrame:
        t = res["scores"][["brier", "brier_ci_low", "brier_ci_high", "skill_vs_base_rate",
                           "skill_ci_low", "skill_ci_high"]]  # fmt: skip
        return t

    body = f"""
## Data

| Dataset | Use | Terms |
|---|---|---|
{data_rows}

Canonical frames: metres on 105 x 68, centre origin, home attacking +x in every period
(direction inferred from goalkeepers). SkillCorner pitches are rescaled from their real size.
Rows more than 5 m outside the lines are dropped at ingest and counted:

{md(oob_totals, 0)}

Per match:

{md(ingest, 0, index=False)}

### Event label integrity

For every labelled event, the tracked player nearest its start position should be the
credited player. Mismatched player-halves with the ids as published:

{md(swap3, 3, index=False)}

Games {train_games}: {train_mismatches} mismatches.
After the correction in `metrica_events.ID_CORRECTIONS`: {sum(swaps_after.values())} mismatches.
{tuning_sentence}

## Pass detection

Owner = nearest player within a radius, gated by ball speed; a pass is an owner change
between teammates, matched to Metrica PASS events with a tolerance of
{params["pass_tolerance_frames"]} frames. v1 uses a fixed radius; v2 derives the radius per
match from unlabelled kick onsets and gates passes on the ball accelerating away. Both were
tuned on games 1-2 only (train F1 v1 {f(params["v1"]["train_scores"]["f1"])}, v2
{f(params["v2"]["train_scores"]["f1"])}); v2 was frozen before its game-{TEST_GAME} evaluation and
was designed after the v1 failure analysis on published labels.

{md(pass_table, 3, index=False)}

Share of labelled releases with the ball more than 0.5 m from the passer (the evidence that
motivated v2; with published labels game {TEST_GAME} looked different, with corrected labels it
does not):

{md(release, 3)}

Remaining failure modes (v2, corrected labels):

{md(failures, 0, index=False)}

## Team shape (median of 5-minute medians)

{md(sn["shape_by_phase"], 1)}

## Pass networks

Inferred networks (from tracking) vs labelled-event networks on game {TEST_GAME}, over the union
of directed player pairs:

{md(sim3, 3, index=False)}

With published labels the home network agreed at Pearson {f(home_v1.loc["raw", "pearson"])};
with corrected labels, {f(home_v1.loc["corrected", "pearson"])}. Largest remaining home edge
residuals (v2 minus corrected events):

{md(residual_table, 0, index=False)}

![Game 3 home pass networks](../../docs/img/phase1/pass_networks_game3_home.png)

## Formations and roles

Mean centroid-relative shape of the 10 outfield players per team, phase, and 5-minute window,
matched to seven templates by assignment cost. `margin` = runner-up cost minus best cost (the
primary ambiguity signal); `relative_margin` = margin / runner-up cost. Neither is a
probability.

**No formation ground truth exists in Metrica** (anonymised, no labels), so Metrica gets
label-free diagnostics only. Roles are checked against SkillCorner's listed positions (one
position per player per match), on the same player-windows for every method:

{md(agree, 3)}

Match-level bootstrap ({BOOTSTRAP_RESAMPLES} resamples), difference and 95% CI:

{md(boot, 3)}

Stability on SkillCorner (share of consecutive windows unchanged), median by phase:

{md(stability_table, 3)}

Metrica, label-free: median margin {f(md_["margin_quantiles"].loc[0.5])}; label and back-line
stability per team and phase:

{md(metrica_stability, 2, index=False)}

Away 3-5-2 vs 5-3-2: {md_["away_352_narrow"]} of {md_["away_352_windows"]} away 3-5-2 windows are
closer to 5-3-2 than the all-window median margin, and {md_["away_352_532_flips"]} of
{md_["away_label_changes"]} away label changes are between the two. {narrow_sentence}

Side agreement by half: {f(side_all.loc[1])} (first) vs {f(side_all.loc[2])} (second) for all
players; {f(side_full.loc[1, "side_template_ok"])} vs {f(side_full.loc[2, "side_template_ok"])} for
players tracked in at least {int(FULL_MATCH_PRESENCE * 100)}% of both halves. No half falls
below 0.5 (minimum {f(sc["side_by_half"]["side_template_ok"].min())}). **Open observation:** the
second-half dip persists for full-match players, but they can still shift roles when
teammates are substituted, so substitutions are not ruled out.

![Formation snapshots](../../docs/img/phase1/formations_game3.png)

## Passing options (pitch control)

Spearman pitch control with LaurieOnTracking's published parameters (nothing fitted),
evaluated only at target points. Baselines fitted on games 1-2: the completion rate (for the
Brier skill score) and logistic regression on pass length and nearest-defender distance to
the target. Uncertainty: block bootstrap over 1-minute blocks.

### (a) Labelled pass attempts (headline)

Completed = PASS; failed = BALL LOST with subtype containing {", ".join(FAILED_PASS_MARKERS)}.
Train {a["n_train"]} attempts (completion {f(a["completion_rate_train"])}), test {a["n_test"]}
(completion {f(a["completion_rate_test"])}).

{md(a_sc, 4)}

Raw pitch control stays the model output; the isotonic calibrator (fitted on games 1-2,
`eval/calibration_phase1.json`) is the display probability for later phases. Note: isotonic
maps {len(iso_zero)} game-{TEST_GAME} attempts to exactly 0, of which
{int(iso_zero["completed"].sum())} were completed; Platt scaling avoids hard zeros at nearly the
same Brier.

![Reliability diagrams](../../docs/img/phase1/reliability_level_a.png)

Failed-pass rule sensitivity (baselines refitted on games 1-2 under each rule):

{md(sens, 4, index=False)}

The "plus ball out" rule adds {", ".join(f"{v} (game {g})" for g, v in ball_out_added.items())}
failed passes: Metrica logs BALL OUT after the recovering team's event, so almost no BALL
LOST is directly followed by it.

### (b) End to end, v2-inferred passes

Completed = v2 pass; failed = v2 turnover with a ball release. Composition on game {TEST_GAME},
by the labelled event nearest each release (labels describe level (b) only):

{md(comp, 3)}

All inferred attempts (completion {f(b_all["completion_rate_test"])}):

{md(score_table(b_all), 4)}

Without tackles and lost dribbles (completion {f(b_clean["completion_rate_test"])}):

{md(score_table(b_clean), 4)}

Level (b) targets are ball positions at receipt, next to whoever got the ball, so it is an
easier task than scoring intended targets, and part of its "failures" are detector errors
({int(comp.loc["labelled completed pass", "attempts"])} attempts match a labelled completed pass).

![Passing options example](../../docs/img/phase1/passing_options_example.png)

## Known limitations

- One held-out game; games 1-2 for all tuning and fitting. Game 3 comes in a different
  provider format (FIFA EPTS) from games 1-2.
- Game-{TEST_GAME} event ids were corrected (P3573/P3580, second half); published-label results are
  shown alongside.
- v2 was designed after seeing v1's game-{TEST_GAME} failures, which were mostly the label swap;
  its gain over v1 is small and may be within noise.
- Failed-pass end locations are interception points, not intended targets: pitch control is
  biased low for failures, which flatters its discrimination.
- Passing-option values for teammates who were not passed to cannot be validated.
- Pitch-control parameters are published defaults; raw values are underconfident in the
  upper range (see reliability diagram) and are model outputs, not calibrated probabilities.
- Metrica has no formation or position labels; SkillCorner lists one position per player
  per match, and wingers are ambiguous between MID and FWD.
- SkillCorner tracking comes from broadcast video; off-camera players are extrapolated by the
  provider.
- Rows beyond 5 m outside the lines are dropped (counts above).

## Reproduce

```
uv sync --group dev
uv run python eval/phase1.py
```

Downloads the open data on first run. Parameters: `eval/params_phase1.json` (re-derived and
checked by this script); calibrators: `eval/calibration_phase1.json`.

## Credits

Tracking and event data: Metrica Sports sample data
(https://github.com/metrica-sports/sample-data). Broadcast tracking: SkillCorner open data
(https://github.com/SkillCorner/opendata, MIT). Pitch control adapted from
LaurieOnTracking (MIT). Loading via kloppy (BSD-3-Clause); plots with mplsoccer (MIT). See
`THIRD_PARTY_NOTICES.md`.
"""
    header = (
        f"# Regista Phase 1 report\n\nGenerated by `eval/phase1.py` on {date.today()}. "
        f"Every number below is computed by that script; none is typed by hand.\n\n"
    )
    return header + tldr + body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-tuning-check", action="store_true",
                        help="do not re-run tuning (saves about 6 minutes)")  # fmt: skip
    args = parser.parse_args()
    report = build_report(args.skip_tuning_check)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"wrote {REPORT_PATH} and figures in {IMG_DIR}")


if __name__ == "__main__":
    main()
