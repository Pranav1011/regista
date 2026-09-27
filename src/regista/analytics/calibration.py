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


def mean_log_loss(w: np.ndarray, design: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean logistic loss and its gradient for weights ``w`` on a design matrix."""
    logits = design @ w
    loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
    grad = design.T @ (1.0 / (1.0 + np.exp(-logits)) - y) / len(y)
    return loss, grad


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Isotonic:
    """Non-decreasing fit of outcomes ``y`` (0/1) on scores ``p``.

    Equal scores are pooled first, so tied samples always share one value and
    knot positions are strictly increasing.
    """
    scores, inverse = np.unique(np.asarray(p, float), return_inverse=True)
    y_sum = np.bincount(inverse, weights=np.asarray(y, float))
    n = np.bincount(inverse).astype(float)
    # pool-adjacent-violators over unique scores: blocks of (sum, count, first, last score)
    sums: list[float] = []
    counts: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for xv, s, c in zip(scores, y_sum, n, strict=True):
        sums.append(s)
        counts.append(c)
        lo.append(xv)
        hi.append(xv)
        while len(sums) > 1 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
            s_last, c_last, h_last = sums.pop(), counts.pop(), hi.pop()
            lo.pop()
            sums[-1] += s_last
            counts[-1] += c_last
            hi[-1] = h_last
    knots_x: list[float] = []
    knots_y: list[float] = []
    for s, c, a, b in zip(sums, counts, lo, hi, strict=True):
        knots_x += [a, b] if b > a else [a]
        knots_y += [s / c, s / c] if b > a else [s / c]
    return Isotonic([float(v) for v in knots_x], [float(v) for v in knots_y])


def fit_platt(p: np.ndarray, y: np.ndarray) -> Platt:
    """Maximum-likelihood Platt scaling."""
    z, y = _logit(np.asarray(p, float)), np.asarray(y, float)
    design = np.column_stack([z, np.ones_like(z)])
    res = minimize(mean_log_loss, np.array([1.0, 0.0]), args=(design, y), jac=True,
                   method="L-BFGS-B")  # fmt: skip
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
