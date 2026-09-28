"""Pressing sanity check on Metrica games 1-3. This is NOT an accuracy measure.

Pressing has no ground truth. As a sanity check only, per 5-minute window and
pressing team, press intensity (share of opponent-carrier frames with a
defender within 5 yd) is correlated with the number of CHALLENGE and RECOVERY
events that team made in the window. Thresholds are fixed a priori (DESIGN.md
ADR-007) and nothing here is tuned.

Run: uv run python eval/pressing_sanity.py
"""

from __future__ import annotations

import pandas as pd
from _common import TEST_GAME, TRAIN_GAMES, load_params, metrica_game
from scipy.stats import spearmanr

from regista.analytics.pressing import press_windows, pressure_frames
from regista.pipeline import v2_owner

WINDOW_S = 300.0
EVENT_TYPES = ("CHALLENGE", "RECOVERY")


def game_windows(game: int, params: dict) -> pd.DataFrame:
    frames, events = metrica_game(game)
    owner, _ = v2_owner(frames, params)
    w = press_windows(pressure_frames(frames, owner), WINDOW_S)
    w = w[w["third"] == "all"].copy()
    t = frames.drop_duplicates(["period", "frame"]).set_index(["period", "frame"])["t"]
    ev = events[events["type"].isin(EVENT_TYPES)].dropna(subset=["start_frame"])
    ev = ev.assign(t=t.reindex(pd.MultiIndex.from_arrays(
        [ev["period"], ev["start_frame"].astype(int)])).to_numpy())  # fmt: skip
    counts = []
    for r in w.itertuples():
        sel = ev[(ev["period"] == r.period) & (ev["team"] == r.pressing_team)
                 & (ev["t"] >= r.t_start) & (ev["t"] < r.t_end)]  # fmt: skip
        counts.append(len(sel))
    w["defensive_events"] = counts
    w["game"] = game
    return w


def evaluate() -> dict:
    params = load_params()
    w = pd.concat([game_windows(g, params) for g in (*TRAIN_GAMES, TEST_GAME)], ignore_index=True)
    w = w[w["complete"] & (w["n_frames"] > 0)]
    rows = []
    for name, g in [*((f"game {k}", v) for k, v in w.groupby("game")), ("pooled", w)]:
        rho = spearmanr(g["press_intensity"], g["defensive_events"])
        rows.append({"scope": name, "windows": len(g), "spearman_rho": rho.statistic,
                     "p_value": rho.pvalue})  # fmt: skip
    summary = w.groupby(["game", "pressing_team"])[
        ["press_intensity", "tight_intensity", "mean_defenders_within"]
    ].median()
    return {"correlation": pd.DataFrame(rows), "summary": summary, "windows": w}


def main() -> None:
    r = evaluate()
    print("## Sanity check (not accuracy): press intensity vs CHALLENGE + RECOVERY counts")
    print(r["correlation"].round(3).to_markdown(index=False))
    print("\n## Median per-window pressing by game and team")
    print(r["summary"].round(3).to_markdown())


if __name__ == "__main__":
    main()
