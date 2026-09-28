"""Why do line-height shifts fire more on SkillCorner? A diagnostic, not a model change.

SkillCorner tracking comes from broadcast video; players off camera are
extrapolated, and the raw files mark each player position with `is_detected`
(kloppy drops the flag, so the raw files are read here, cached under
data/raw/skillcorner/). Per causal stream window and team, this relates the
5-minute change in out-of-possession line height to the share of that team's
player positions that were detected (not extrapolated) during its
out-of-possession frames. No threshold is changed by this script.

Run: uv run python eval/skillcorner_detection.py
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd
from _common import SKILLCORNER_MATCHES, load_params, skillcorner_match
from scipy.stats import spearmanr
from tune_moments import cached_windows

from regista import io
from regista.pipeline import possession_phase
from regista.schema import Source

RAW_URL = ("https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/"
           "{m}/{m}_tracking_extrapolated.jsonl")  # fmt: skip
LAG_STEPS = 5  # 5 minutes at 1-minute steps


def detection_flags(match_id: str) -> pd.DataFrame:
    """(period, frame, player_id, is_detected) from the raw SkillCorner file."""
    path = io.raw_dir(Source.SKILLCORNER) / f"{match_id}_tracking_extrapolated.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(RAW_URL.format(m=match_id), tmp)
        tmp.rename(path)
    rows = []
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("period") is None:
                continue
            for p in rec.get("player_data") or []:
                rows.append((int(rec["period"]), int(rec["frame"]), str(p["player_id"]),
                             bool(p["is_detected"])))  # fmt: skip
    return pd.DataFrame(rows, columns=["period", "frame", "player_id", "is_detected"])


def match_table(match_id: str, params: dict) -> pd.DataFrame:
    frames, players = skillcorner_match(match_id)
    windows = cached_windows("skillcorner", match_id, lambda: frames, params)
    phase = possession_phase(frames, params).rename(columns={"team": "possession_team"})
    flags = detection_flags(match_id).merge(players[["player_id", "team"]], on="player_id")
    t_of = frames.drop_duplicates(["period", "frame"])[["period", "frame", "t"]]
    flags = flags.merge(t_of, on=["period", "frame"]).merge(
        phase[["period", "frame", "possession_team"]], on=["period", "frame"]
    )
    out_of_poss = flags[
        flags["possession_team"].notna() & (flags["possession_team"] != flags["team"])
    ]

    rows = []
    for (team, period), w in windows[windows["complete"]].groupby(["team", "period"]):
        w = w.sort_values("t_end").reset_index(drop=True)
        f = out_of_poss[(out_of_poss["team"] == team) & (out_of_poss["period"] == period)]
        for i in range(LAG_STEPS, len(w)):
            lo, hi = w["t_start"][i - LAG_STEPS], w["t_end"][i]
            sel = f[(f["t"] >= lo) & (f["t"] < hi)]
            rows.append({
                "match": match_id,
                "team": team,
                "period": period,
                "t_end": w["t_end"][i],
                "abs_line_change_5min": abs(w["line_height_out"][i]
                                            - w["line_height_out"][i - LAG_STEPS]),
                "detected_share": float(sel["is_detected"].mean()) if len(sel) else np.nan,
            })  # fmt: skip
    return pd.DataFrame(rows)


def evaluate() -> dict:
    params = load_params()
    t = pd.concat([match_table(m, params) for m in SKILLCORNER_MATCHES], ignore_index=True)
    t = t.dropna()
    rho = spearmanr(t["detected_share"], t["abs_line_change_5min"])
    t["detected_quartile"] = pd.qcut(t["detected_share"], 4, labels=["Q1 (least)", "Q2", "Q3",
                                                                      "Q4 (most)"])  # fmt: skip
    by_q = t.groupby("detected_quartile", observed=True).agg(
        windows=("abs_line_change_5min", "size"),
        detected_share_median=("detected_share", "median"),
        abs_line_change_5min_median=("abs_line_change_5min", "median"),
    )
    return {"table": t, "spearman": rho, "by_quartile": by_q,
            "detected_share_overall": float(t["detected_share"].median())}  # fmt: skip


def main() -> None:
    r = evaluate()
    print(f"windows: {len(r['table'])}; median share of out-of-possession player positions "
          f"detected: {r['detected_share_overall']:.3f}")  # fmt: skip
    print(f"Spearman(detected share, |5-min line-height change|) = "
          f"{r['spearman'].statistic:.3f} (p = {r['spearman'].pvalue:.3g})")  # fmt: skip
    print(r["by_quartile"].round(3).to_markdown())


if __name__ == "__main__":
    main()
