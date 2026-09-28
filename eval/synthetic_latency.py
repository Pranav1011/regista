"""Detection latency on SYNTHETIC matches with known change times (labelled as synthetic).

Real matches have no known onset, so their "delay" (emit minus estimated start)
is definitional. Here the frozen detectors (eval/params_phase2.json) run on the
synthetic matches from tests/synthetic_match.py with one injected change each,
10 minutes into period 1, and latency is emit time minus the true change time.
These are synthetic known-answer numbers, not measurements of real matches.

Run: uv run python eval/synthetic_latency.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from _common import load_params
from tune_moments import PARAMS2_PATH

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from synthetic_match import MatchSpec, make_match  # noqa: E402

from regista.moments import DetectorConfig, detect_moments, stream_windows  # noqa: E402

CHANGE_T = 600.0
CASES = {
    "back_line_change": ("away", {"away_formation": lambda p, t: "5-3-2" if p == 2 or t >= CHANGE_T
                                  else "4-4-2"}),
    "press_change": ("away", {"away_press": lambda p, t: 1.5 if p == 1 and t < CHANGE_T
                              else None}),
    "line_height_shift": ("home", {"home_line": lambda p, t: 20.0 if p == 2 or t >= CHANGE_T
                                   else 0.0}),
}  # fmt: skip


def evaluate() -> pd.DataFrame:
    params = load_params()
    cfg = DetectorConfig(**json.loads(PARAMS2_PATH.read_text())["detectors"])
    rows = []
    for kind, (team, hooks) in CASES.items():
        frames = make_match(MatchSpec(period_s=1500.0, fps=5.0, **hooks))
        m = detect_moments(stream_windows(frames, params), cfg)
        hit = m[(m["type"] == kind) & (m["team"] == team) & (m["period"] == 1)]
        rows.append({
            "type": kind,
            "detected": len(hit) > 0,
            "latency_min": (hit["emit_t"].iat[0] - CHANGE_T) / 60 if len(hit) else float("nan"),
            "start_estimate_error_min": (hit["start_t"].iat[0] - CHANGE_T) / 60
            if len(hit) else float("nan"),
            "other_alerts_period_1": int(((m["period"] == 1) & (m["type"] != kind)).sum()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    print("SYNTHETIC matches, one injected change each at 10:00 (frozen detectors)")
    print(evaluate().round(2).to_markdown(index=False))


if __name__ == "__main__":
    main()
