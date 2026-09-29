"""Frozen moment detectors on held-out Metrica game 3 and every SkillCorner match.

Thresholds come from eval/params_phase2.json (tuned on games 1-2 only). Reports
alert counts per match and type, detection delay (emit minus estimated start),
and game 3's full alert list. A match above 12 alerts is flagged, never truncated.

Writes eval/reports/phase2_moments.md. Game 3's alert list is checked against
the match store the viewer is exported from, so the report and the viewer show
the same moments.

Run: uv run python eval/moments_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from _common import (
    SKILLCORNER_MATCHES,
    TEST_GAME,
    TRAIN_GAMES,
    load_params,
    metrica_game,
    skillcorner_match,
)
from tune_moments import PARAMS2_PATH, cached_windows

from regista import io
from regista.clock import match_clock
from regista.moments import DetectorConfig, detect_moments
from regista.store import Store

REPORT_CAP = 12  # reporting threshold, not truncation
REPORT = Path(__file__).resolve().parent / "reports" / "phase2_moments.md"
CHECKPOINT_A_NOTE = (
    "The alert list shown at Checkpoint A had a third game-3 alert, an away press change "
    "at 64:00 (severity 1.21), from the detectors before the coverage rule. The coverage "
    "rule (a window needs at least `min_coverage_min` = 4 minutes of in-period data) "
    "makes the 04:00 window at the start of the match usable; it sets the away team's "
    "initial press reference, which carries through the match, so the 59:30-64:00 rise "
    "(0.35 to 0.68) no longer departs from the reference by `press_delta`. With a "
    "5-minute threshold the current code reproduces the 64:00 alert exactly. The "
    "first-half alert shown then as 47:00 is the same moment, now written in stoppage "
    "time (45+2:00; the stoppage clock came in the same commit), with severity 1.148 "
    "then and 1.160 under the coverage rule."
)


def detector_config() -> DetectorConfig:
    return DetectorConfig(**json.loads(PARAMS2_PATH.read_text())["detectors"])


def all_moments() -> dict[tuple[str, str], pd.DataFrame]:
    params, cfg = load_params(), detector_config()
    matches = [("metrica", str(g), lambda g=g: metrica_game(g)[0])
               for g in (*TRAIN_GAMES, TEST_GAME)]  # fmt: skip
    matches += [
        ("skillcorner", m, lambda m=m: skillcorner_match(m)[0]) for m in SKILLCORNER_MATCHES
    ]
    return {(src, mid): detect_moments(cached_windows(src, mid, fn, params), cfg)
            for src, mid, fn in matches}  # fmt: skip


def describe(m: pd.DataFrame) -> pd.DataFrame:
    """Readable alert list: clock, type, team, severity, evidence summary."""
    rows = []
    for r in m.itertuples():
        ev = r.evidence
        before, after = ev["before"], ev["after"]
        if r.type == "back_line_change":
            detail = f"back {int(before)} -> {int(after)} (labels {', '.join(ev['labels'])})"
        elif r.type == "press_change":
            detail = f"press intensity {before:.2f} -> {after:.2f}"
        else:
            detail = f"line height {before:.1f} m -> {after:.1f} m"
        rows.append({"emit_clock": match_clock(r.period, r.emit_t),
                     "start_clock (est.)": match_clock(r.period, r.start_t), "type": r.type,
                     "team": r.team, "severity": r.severity, "evidence": detail})  # fmt: skip
    return pd.DataFrame(rows)


def evaluate() -> dict:
    moments = all_moments()
    counts = []
    for (src, mid), m in moments.items():
        row = {"source": src, "match": mid,
               "split": "design (games 1-2)" if src == "metrica" and int(mid) in TRAIN_GAMES
               else "held out", "total": len(m)}  # fmt: skip
        for kind in ("back_line_change", "press_change", "line_height_shift"):
            row[kind] = int((m["type"] == kind).sum())
        row["above_cap"] = len(m) > REPORT_CAP
        counts.append(row)
    counts = pd.DataFrame(counts)
    held = pd.concat([m for (src, mid), m in moments.items()
                      if not (src == "metrica" and int(mid) in TRAIN_GAMES)])  # fmt: skip
    delay = (held["emit_t"] - held["start_t"]) / 60
    return {
        "counts": counts,
        "delay_min": delay.describe(percentiles=[0.5, 0.9]),
        "delay_by_type": (
            held.assign(delay=delay).groupby("type")["delay"].describe(percentiles=[0.5, 0.9])
        ),  # fmt: skip
        "game3": describe(moments[("metrica", str(TEST_GAME))]),
        "moments": moments,
    }


def check_against_store(game3: pd.DataFrame) -> None:
    """Raise if game 3's alerts differ from the match store (the viewer's source)."""
    store = Store(io.data_dir() / "store" / "metrica" / str(TEST_GAME))
    mo = store.table("moments").sort_values(["period", "emit_t"])
    stored = [
        (match_clock(int(r.period), float(r.emit_t)), r.type, r.team) for r in mo.itertuples()
    ]
    listed = list(zip(game3["emit_clock"], game3["type"], game3["team"], strict=True))
    if stored != listed:
        raise RuntimeError(f"game {TEST_GAME} alerts differ from the store: {listed} vs {stored}")


def write_report(r: dict) -> Path:
    held = r["counts"][r["counts"]["split"] == "held out"]
    lines = [
        "# Regista Phase 2: detected moments",
        "",
        "Generated by `eval/moments_eval.py`. Thresholds from `eval/params_phase2.json`, "
        "tuned on Metrica games 1-2 only.",
        "",
        f"## Metrica game {TEST_GAME} alerts (held out)",
        "",
        "Checked against the match store the viewer is exported from.",
        "",
        r["game3"].to_markdown(index=False),
        "",
        CHECKPOINT_A_NOTE,
        "",
        "## Alert counts per match",
        "",
        r["counts"].to_markdown(index=False),
        "",
        f"Held-out matches: {len(held)}; alerts per match median {held['total'].median()}, "
        f"range {held['total'].min()}-{held['total'].max()}; above {REPORT_CAP}: "
        f"{int(held['above_cap'].sum())} (a reporting threshold, never a truncation).",
        "",
        "## Detection delay, minutes (emit minus estimated start), held out",
        "",
        r["delay_by_type"].round(2).to_markdown(),
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    return REPORT


def main() -> None:
    r = evaluate()
    check_against_store(r["game3"])
    print(f"wrote {write_report(r)}")
    print(f"## Game {TEST_GAME} alerts (held out)")
    print(r["game3"].to_markdown(index=False))
    print("\n## Alert counts per match")
    print(r["counts"].to_markdown(index=False))
    held = r["counts"][r["counts"]["split"] == "held out"]
    print(f"\nheld-out matches: {len(held)}, alerts per match median {held['total'].median()}, "
          f"range {held['total'].min()}-{held['total'].max()}, above {REPORT_CAP}: "
          f"{int(held['above_cap'].sum())}")  # fmt: skip
    print("\n## Detection delay, minutes (emit minus estimated start), held out")
    print(r["delay_by_type"].round(2).to_markdown())


if __name__ == "__main__":
    main()
