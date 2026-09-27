"""Post-hoc calibration of model probabilities: isotonic regression and Platt scaling.

The raw model output (e.g. pitch control) stays the model's output. A fitted
calibrator maps it to a display probability, and is fitted on training data only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

EPS = 1e-6


@dataclass
class Isotonic:
    """Monotone step function fitted by pool-adjacent-violators; linear between knots."""

    x: list[float]  # knot positions (increasing)
    y: list[float]  # calibrated values at the knots (non-decreasing)

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(p, dtype=float), self.x, self.y)

    def to_dict(self) -> dict:
        return {"kind": "isotonic", **asdict(self)}


@dataclass
class Platt:
    """Logistic regression on the logit of the raw probability."""

    a: float
    b: float

    def predict(self, p: np.ndarray) -> np.ndarray:
        z = _logit(np.asarray(p, dtype=float))
        return 1.0 / (1.0 + np.exp(-(self.a * z + self.b)))

    def to_dict(self) -> dict:
        return {"kind": "platt", **asdict(self)}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Isotonic:
    """Non-decreasing fit of outcomes ``y`` (0/1) on scores ``p``."""
    order = np.argsort(p, kind="stable")
    xs, ys = np.asarray(p, float)[order], np.asarray(y, float)[order]
    # pool-adjacent-violators on blocks of (sum, count, x range)
    sums, counts, lo, hi = [], [], [], []
    for xv, yv in zip(xs, ys, strict=True):
        sums.append(yv)
        counts.append(1.0)
        lo.append(xv)
        hi.append(xv)
        while len(sums) > 1 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
            s, c, h = sums.pop(), counts.pop(), hi.pop()
            lo.pop()
            sums[-1] += s
            counts[-1] += c
            hi[-1] = h
    knots_x, knots_y = [], []
    for s, c, a, b in zip(sums, counts, lo, hi, strict=True):
        value = s / c
        knots_x += [a, b] if b > a else [a]
        knots_y += [value, value] if b > a else [value]
    return Isotonic([float(v) for v in knots_x], [float(v) for v in knots_y])


def fit_platt(p: np.ndarray, y: np.ndarray) -> Platt:
    """Maximum-likelihood Platt scaling."""
    z, y = _logit(np.asarray(p, float)), np.asarray(y, float)

    def nll(w: np.ndarray) -> float:
        logits = w[0] * z + w[1]
        return float(np.sum(np.logaddexp(0.0, logits) - y * logits))

    res = minimize(nll, np.array([1.0, 0.0]), method="BFGS")
    if not res.success:
        raise RuntimeError(f"Platt scaling did not converge: {res.message}")
    return Platt(float(res.x[0]), float(res.x[1]))


def from_dict(d: dict) -> Isotonic | Platt:
    kind = d.get("kind")
    if kind == "isotonic":
        return Isotonic(d["x"], d["y"])
    if kind == "platt":
        return Platt(d["a"], d["b"])
    raise ValueError(f"unknown calibrator kind {kind!r}")
