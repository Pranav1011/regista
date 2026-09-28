"""Frozen moment detectors on held-out Metrica game 3 and every SkillCorner match.

Thresholds come from eval/params_phase2.json (tuned on games 1-2 only). Reports
alert counts per match and type, detection delay (emit minus estimated start),
and game 3's full alert list. A match above 12 alerts is flagged, never truncated.

Run: uv run python eval/moments_eval.py
"""

from __future__ import annotations

import json

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

from regista.clock import match_clock
from regista.moments import DetectorConfig, detect_moments

REPORT_CAP = 12  # reporting threshold, not truncation


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


def main() -> None:
    r = evaluate()
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
