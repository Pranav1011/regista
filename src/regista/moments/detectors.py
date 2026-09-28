"""Moment detectors over causal trailing-window features (``engine.stream_windows``).

Each detector keeps a reference state per team and fires when the window value
departs from it consistently for ``persistence_min`` minutes (consecutive
1-minute steps within one period); the reference then moves to the new state.
The reference carries over half-time, so a change at the break is detected
early in the second half. Only complete
windows (a full ``window_s`` of data) are used.

- back_line_change: out-of-possession back-line count differs from the
  reference, every window in the run has a template margin of at least
  ``min_margin``, and the new count is the same throughout the run.
- press_change: press intensity differs from the reference by more than
  ``press_delta``, in the same direction throughout the run.
- line_height_shift: out-of-possession line height differs from the reference
  by more than ``line_delta_m`` metres, in the same direction throughout.

``start_t`` is an estimate of when the change began: a trailing window reflects
a new state once about half of it lies after the change, so start_t is the end
of the first differing window minus half a window. ``emit_t`` is the end of the
window that completed the persistence run. After an alert, a detector is
refractory for one window length: a trailing window keeps changing while the
new state fills it, so the reference follows the value without firing, and
one transition gives one alert. Severity is the size of the change
relative to its threshold (for back-line changes: count change times the
smallest margin in the run relative to ``min_margin``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

MOMENT_TYPES = ("back_line_change", "press_change", "line_height_shift")


@dataclass(frozen=True)
class DetectorConfig:
    persistence_min: int = 3
    min_margin: float = 0.03
    press_delta: float = 0.12
    line_delta_m: float = 5.0
    window_s: float = 300.0
    step_s: float = 60.0


@dataclass
class Moment:
    type: str
    team: str
    match_id: str
    period: int
    start_t: float
    emit_t: float
    emit_frame: int
    severity: float
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class _Persistent:
    """Reference-and-run state machine shared by all detectors."""

    def __init__(self, kind: str, team: str, cfg: DetectorConfig):
        self.kind, self.team, self.cfg = kind, team, cfg
        self.reference: float | None = None
        self.reference_window: tuple | None = None
        self.run: list[dict] = []
        self.refractory_until: tuple | None = None  # (period, t_end)

    def _differs(self, w: dict) -> int:
        """+1 / -1 if the window departs from the reference upwards/downwards, 0 if not."""
        raise NotImplementedError

    def _value(self, w: dict) -> float:
        raise NotImplementedError

    def _usable(self, w: dict) -> bool:
        return not np.isnan(self._value(w))

    def _severity(self, run: list[dict]) -> float:
        raise NotImplementedError

    def update(self, w: dict) -> Moment | None:
        if not self._usable(w):
            self.run = []
            return None
        if self.reference is None or (
            self.refractory_until is not None and (w["period"], w["t_end"]) <= self.refractory_until
        ):
            self.reference = self._value(w)
            self.reference_window = (w["period"], w["t_start"], w["t_end"])
            self.run = []
            return None
        if self.run and w["period"] != self.run[0]["period"]:
            self.run = []  # runs never span half-time; the reference carries over
        direction = self._differs(w)
        if direction == 0 or (self.run and direction != self.run[0]["_dir"]):
            self.run = [] if direction == 0 else [{**w, "_dir": direction}]
            return None
        self.run.append({**w, "_dir": direction})
        if len(self.run) < self.cfg.persistence_min:
            return None
        first, last = self.run[0], self.run[-1]
        before = self.reference
        after = float(np.median([self._value(r) for r in self.run]))
        moment = Moment(
            type=self.kind,
            team=self.team,
            match_id=w["match_id"],
            period=int(first["period"]),
            start_t=max(first["t_end"] - self.cfg.window_s / 2, 0.0),
            emit_t=float(last["t_end"]),
            emit_frame=int(last["emit_frame"]),
            severity=round(self._severity(self.run), 3),
            evidence={
                "before": before,
                "after": after,
                "reference_window": list(self.reference_window),
                "run_windows": [[int(r["period"]), r["t_start"], r["t_end"]] for r in self.run],
                **self._extra_evidence(self.run),
            },
        )
        self.reference = self._value(last)
        self.reference_window = (last["period"], last["t_start"], last["t_end"])
        self.run = []
        self.refractory_until = (last["period"], last["t_end"] + self.cfg.window_s)
        return moment

    def _extra_evidence(self, run: list[dict]) -> dict:
        return {}


class _BackLine(_Persistent):
    def _value(self, w):
        return w["back_line_out"]

    def _usable(self, w):
        return not np.isnan(w["back_line_out"]) and w["margin_out"] >= self.cfg.min_margin

    def _differs(self, w):
        if w["back_line_out"] == self.reference:
            return 0
        if self.run and w["back_line_out"] != self.run[0]["back_line_out"]:
            return 0  # a different new count: not the same change
        return 1 if w["back_line_out"] > self.reference else -1

    def _severity(self, run):
        return abs(run[-1]["back_line_out"] - self.reference) * (
            min(r["margin_out"] for r in run) / self.cfg.min_margin
        )

    def _extra_evidence(self, run):
        return {"labels": [r["label_out"] for r in run],
                "margins": [round(r["margin_out"], 4) for r in run]}  # fmt: skip


class _Threshold(_Persistent):
    column: str
    delta_attr: str

    def _value(self, w):
        return w[self.column]

    def _differs(self, w):
        d = w[self.column] - self.reference
        threshold = getattr(self.cfg, self.delta_attr)
        return 0 if abs(d) <= threshold else (1 if d > 0 else -1)

    def _severity(self, run):
        after = float(np.median([r[self.column] for r in run]))
        return abs(after - self.reference) / getattr(self.cfg, self.delta_attr)

    def _extra_evidence(self, run):
        return {"values": [round(r[self.column], 4) for r in run]}


class _Press(_Threshold):
    column, delta_attr = "press_intensity", "press_delta"


class _Line(_Threshold):
    column, delta_attr = "line_height_out", "line_delta_m"


_DETECTORS = {"back_line_change": _BackLine, "press_change": _Press, "line_height_shift": _Line}


def detect_moments(windows: pd.DataFrame, config: DetectorConfig | None = None) -> pd.DataFrame:
    """Run every detector over stream windows in time order; returns one row per moment."""
    cfg = config or DetectorConfig()
    complete = windows[windows["complete"]].sort_values(["period", "t_end", "team"])
    teams = complete["team"].unique()
    state = {(kind, team): cls(kind, team, cfg) for kind, cls in _DETECTORS.items()
             for team in teams}  # fmt: skip
    moments = []
    for w in complete.to_dict("records"):
        for kind in _DETECTORS:
            m = state[(kind, w["team"])].update(w)
            if m is not None:
                moments.append(m.to_dict())
    cols = ["type", "team", "match_id", "period", "start_t", "emit_t", "emit_frame", "severity",
            "evidence"]  # fmt: skip
    return pd.DataFrame(moments, columns=cols)
