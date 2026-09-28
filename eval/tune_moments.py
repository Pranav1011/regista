"""Freeze moment-detector thresholds on Metrica games 1-2 ONLY.

Persistence is fixed at 3 minutes from eval/rolling_stability.py (a third of
out-of-possession back-line runs last 2 minutes or less: flicker). Each
detector's threshold is then chosen so that its mean alert count on games 1-2
is closest to 2 per match (about 6 alerts per match in total, inside the 4-8
target); ties go to fewer alerts, then to the smallest threshold achieving them.
Then the minimum in-period coverage N (minutes of data a window needs before
it can trigger, e.g. just after a restart) is set by rule, with the thresholds
held fixed: the smallest N whose windows are no noisier than full windows
(median step-to-step change in press intensity and line height, and back-line
flip rate, each within 10% of the full-window value). The threshold search uses
the rule it was frozen under (windows ending at least one window length into the
period, `complete`), so adding N afterwards does not re-tune the thresholds.
Writes eval/params_phase2.json.

Run: uv run python eval/tune_moments.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pandas as pd
from _common import EVAL_DIR, TRAIN_GAMES, load_params, metrica_game

from regista import io
from regista.moments import DetectorConfig, StreamConfig, detect_moments, stream_windows

PARAMS2_PATH = EVAL_DIR / "params_phase2.json"
SENSITIVITY_PATH = EVAL_DIR / "reports" / "phase2_threshold_sensitivity.json"
IMG_DIR = EVAL_DIR.parent / "docs" / "img" / "phase2"
PERSISTENCE_MIN = 3
TARGET_PER_TYPE = 2.0
GRIDS = {
    "back_line_change": ("min_margin", (0.011, 0.027, 0.055, 0.08)),
    "press_change": (
        "press_delta",
        (0.10, 0.15, 0.20, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.35, 0.40),
    ),
    "line_height_shift": ("line_delta_m", (5.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0)),
}


def cached_windows(source: str, match_id: str, frames_fn, params: dict) -> pd.DataFrame:
    """Stream windows for a match, cached under data/cache (recomputed if missing)."""
    path = io.data_dir() / "cache" / "stream_windows" / source / f"{match_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    w = stream_windows(frames_fn(), params, StreamConfig())
    path.parent.mkdir(parents=True, exist_ok=True)
    w.to_parquet(path, index=False)
    return w


FULL_WINDOW_MIN = 5.0
NOISE_TOLERANCE = 1.10


def coverage_noise(windows: dict) -> tuple[pd.DataFrame, float]:
    """Step-to-step noise by window coverage (minutes) on the design games; the chosen N."""
    rows = []
    for w in windows.values():
        for _, h in w.groupby(["team", "period"]):
            h = h.sort_values("t_end").reset_index(drop=True)
            for i in range(1, len(h)):
                rows.append({
                    "coverage_min": round(h["coverage_s"][i] / 60),
                    "d_press": abs(h["press_intensity"][i] - h["press_intensity"][i - 1]),
                    "d_line": abs(h["line_height_out"][i] - h["line_height_out"][i - 1]),
                    "flip": float(h["back_line_out"][i] != h["back_line_out"][i - 1]),
                })  # fmt: skip
    d = pd.DataFrame(rows)
    table = d.groupby("coverage_min").agg(windows=("flip", "size"), d_press=("d_press", "median"),
                                          d_line=("d_line", "median"),
                                          flip_rate=("flip", "mean"))  # fmt: skip
    full = table.loc[FULL_WINDOW_MIN]
    ok = (table[["d_press", "d_line", "flip_rate"]] <= full[["d_press", "d_line", "flip_rate"]]
          * NOISE_TOLERANCE).all(axis=1)  # fmt: skip
    candidates = [n for n in table.index if n <= FULL_WINDOW_MIN and ok.loc[n]
                  and all(ok.loc[m] for m in table.index if n <= m <= FULL_WINDOW_MIN)]  # fmt: skip
    return table.assign(no_noisier_than_full=ok), float(min(candidates))


def plot_press_sensitivity(grid: pd.DataFrame, chosen: float) -> None:
    """Alert rate vs press-change threshold on the design games (for DESIGN.md)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = grid[grid["type"] == "press_change"].sort_values("press_delta")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(g["press_delta"], g["mean"], marker="o", color="#1f5f99")
    ax.axhline(TARGET_PER_TYPE, color="#999", ls="--", lw=1, label="target (2 per match)")
    ax.axvline(chosen, color="#c8553d", lw=1, label=f"frozen threshold {chosen:.2f}")
    ax.set_xlabel("press-change threshold (change in press intensity)")
    ax.set_ylabel("press alerts per match")
    ax.set_title("Press-change alert rate vs threshold, Metrica games 1-2 (design)", fontsize=9)
    ax.legend(fontsize=8)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "press_threshold_sensitivity.png", dpi=110)
    plt.close(fig)


def main() -> None:
    params = load_params()
    windows = {g: cached_windows("metrica", str(g), lambda g=g: metrica_game(g)[0], params)
               for g in TRAIN_GAMES}  # fmt: skip
    base = DetectorConfig(persistence_min=PERSISTENCE_MIN, min_coverage_min=0.0)
    search = {
        g: w[w["complete"]] for g, w in windows.items()
    }  # the rule thresholds were frozen under
    chosen, table = {}, []
    for kind, (attr, grid) in GRIDS.items():
        best = None
        for value in grid:
            cfg = replace(base, **{attr: value})
            counts = [int((detect_moments(w, cfg)["type"] == kind).sum()) for w in search.values()]
            mean = sum(counts) / len(counts)
            table.append({"type": kind, attr: value, "counts": counts, "mean": mean})
            key = (abs(mean - TARGET_PER_TYPE), mean, value)
            if best is None or key < best[0]:
                best = (key, value)
        chosen[attr] = best[1]
    noise, coverage_min = coverage_noise(windows)
    frozen = replace(base, **chosen, min_coverage_min=coverage_min)
    totals = [len(detect_moments(w, frozen)) for w in windows.values()]
    out = {
        "tuned_on": list(TRAIN_GAMES),
        "stream": asdict(StreamConfig()),
        "detectors": asdict(frozen),
        "target_per_type": TARGET_PER_TYPE,
        "coverage_noise": noise.reset_index().to_dict("records"),
        "train_alerts_per_match": dict(zip(map(str, TRAIN_GAMES), totals, strict=True)),
    }
    PARAMS2_PATH.write_text(json.dumps(out, indent=2) + "\n")
    grid = pd.DataFrame(table)
    SENSITIVITY_PATH.write_text(grid.to_json(orient="records", indent=2) + "\n")
    plot_press_sensitivity(grid, chosen["press_delta"])
    print(pd.DataFrame(table).to_markdown(index=False))
    print(noise.round(3).to_markdown())
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
