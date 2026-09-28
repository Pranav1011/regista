"""Formation stability on rolling windows (5-min windows, 1-min steps), Metrica games 1-2.

Used to choose the detectors' persistence (in minutes), so it runs on the
design games only. With 1-minute steps consecutive windows overlap by 80%, so
"unchanged from the previous step" is much higher than for tiled windows; the
run-length distribution (how many minutes a back-line state lasts) is the
relevant quantity.

Run: uv run python eval/rolling_stability.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import TRAIN_GAMES, load_params, metrica_game

from regista.analytics.formations import (
    back_line,
    detect_formations,
    label_stability,
    window_shapes,
)
from regista.pipeline import possession_phase

WINDOW_S, STEP_S = 300.0, 60.0


def run_lengths(formations: pd.DataFrame, key) -> pd.Series:
    """Lengths (in steps = minutes) of runs of an unchanged value, per team and phase."""
    out = []
    for _, g in formations.dropna(subset=["label"]).groupby(["match_id", "team", "phase"]):
        vals = [key(v) for v in g.sort_values(["period", "window"])["label"]]
        run = 1
        for a, b in zip(vals, vals[1:], strict=False):
            if a == b:
                run += 1
            else:
                out.append(run)
                run = 1
        out.append(run)
    return pd.Series(out)


def evaluate() -> dict:
    params = load_params()
    forms = []
    for game in TRAIN_GAMES:
        frames, _ = metrica_game(game)
        f, _ = detect_formations(window_shapes(frames, possession_phase(frames, params),
                                               WINDOW_S, STEP_S))  # fmt: skip
        forms.append(f)
    formations = pd.concat(forms, ignore_index=True)
    identity = lambda v: v  # noqa: E731
    return {
        "label_stability": label_stability(formations).groupby("phase")["stability"].median(),
        "back_line_stability": label_stability(formations, key=back_line)
        .groupby("phase")["stability"]
        .median(),
        "label_runs": run_lengths(formations, identity),
        "back_line_runs": run_lengths(formations[formations["phase"] == "out"], back_line),
        "margins": formations["margin"].dropna(),
        "formations": formations,
    }


def main() -> None:
    r = evaluate()
    print("step-to-step stability (median over team x phase):")
    print("  label", r["label_stability"].round(3).to_dict())
    print("  back line", r["back_line_stability"].round(3).to_dict())
    for name in ("label_runs", "back_line_runs"):
        runs = r[name]
        print(f"{name} (minutes, out-of-possession for back line): n={len(runs)}, "
              f"quantiles {runs.quantile([0.25, 0.5, 0.75, 0.9]).to_dict()}, "
              f"share of runs <= 2 min {np.mean(runs <= 2):.3f}")  # fmt: skip
    print(
        "rolling-window margin quantiles:",
        r["margins"].quantile([0.1, 0.25, 0.5]).round(3).to_dict(),
    )


if __name__ == "__main__":
    main()
