"""Known-answer tests for calibration. All data here is synthetic, by design."""

import numpy as np
import pytest

from regista.analytics.calibration import fit_isotonic, fit_platt, from_dict


def test_isotonic_pools_violators_into_block_means():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    iso = fit_isotonic(p, y)
    # (0.2, 0.3) violate monotonicity and pool to 0.5
    np.testing.assert_allclose(iso.predict(p), [0.0, 0.5, 0.5, 1.0])
    assert np.all(np.diff(iso.predict(np.linspace(0, 1, 50))) >= 0)


def test_isotonic_recovers_a_miscalibrated_score():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(size=true_p.size) < true_p).astype(float)
    score = true_p**3  # monotone but badly calibrated
    iso = fit_isotonic(score, y)
    for q in (0.1, 0.5, 0.9):
        assert iso.predict(np.array([q**3]))[0] == pytest.approx(q, abs=0.05)


def test_platt_recovers_logistic_distortion():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 20000)
    y = (rng.uniform(size=z.size) < 1 / (1 + np.exp(-(0.5 * z + 1.0)))).astype(float)
    score = 1 / (1 + np.exp(-z))
    platt = fit_platt(score, y)
    assert (platt.a, platt.b) == pytest.approx((0.5, 1.0), abs=0.1)


def test_calibrators_roundtrip_through_dict():
    iso = fit_isotonic(np.array([0.1, 0.9]), np.array([0.0, 1.0]))
    assert from_dict(iso.to_dict()).predict(np.array([0.5]))[0] == pytest.approx(0.5)
    platt = fit_platt(np.array([0.2, 0.4, 0.6, 0.8] * 5), np.array([0, 0, 1, 1] * 5))
    np.testing.assert_allclose(from_dict(platt.to_dict()).predict([0.3]), platt.predict([0.3]))
