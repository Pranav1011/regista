"""Freeze moment-detector thresholds on Metrica games 1-2 ONLY.

Persistence is fixed at 3 minutes from eval/rolling_stability.py (a third of
out-of-possession back-line runs last 2 minutes or less: flicker). Each
detector's threshold is then chosen so that its mean alert count on games 1-2
is closest to 2 per match (about 6 alerts per match in total, inside the 4-8
target); ties go to fewer alerts, then to the smallest threshold achieving them.
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


def main() -> None:
    params = load_params()
    windows = {g: cached_windows("metrica", str(g), lambda g=g: metrica_game(g)[0], params)
               for g in TRAIN_GAMES}  # fmt: skip
    base = DetectorConfig(persistence_min=PERSISTENCE_MIN)
    chosen, table = {}, []
    for kind, (attr, grid) in GRIDS.items():
        best = None
        for value in grid:
            cfg = replace(base, **{attr: value})
            counts = [int((detect_moments(w, cfg)["type"] == kind).sum()) for w in windows.values()]
            mean = sum(counts) / len(counts)
            table.append({"type": kind, attr: value, "counts": counts, "mean": mean})
            key = (abs(mean - TARGET_PER_TYPE), mean, value)
            if best is None or key < best[0]:
                best = (key, value)
        chosen[attr] = best[1]
    frozen = replace(base, **chosen)
    totals = [len(detect_moments(w, frozen)) for w in windows.values()]
    out = {
        "tuned_on": list(TRAIN_GAMES),
        "stream": asdict(StreamConfig()),
        "detectors": asdict(frozen),
        "target_per_type": TARGET_PER_TYPE,
        "train_alerts_per_match": dict(zip(map(str, TRAIN_GAMES), totals, strict=True)),
    }
    PARAMS2_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(pd.DataFrame(table).to_markdown(index=False))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
