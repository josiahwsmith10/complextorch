"""Tests for the numerically-stable complex-VD KL penalty."""

from __future__ import annotations

import numpy as np
import scipy.special
import torch

from complextorch.nn.relevance import LinearVD
from complextorch.nn.relevance.linear import _log_uniform_vd_penalty


def _reference(t: np.ndarray) -> np.ndarray:
    """Float64 exact formula: gamma + t - Ei(-e^t)."""
    return np.euler_gamma + t - scipy.special.expi(-np.exp(t))


def test_vd_penalty_matches_reference_exact_branch():
    t = torch.tensor([-4.9, -3.0, -1.0, 0.0, 2.0, 5.0], dtype=torch.float64)
    out = _log_uniform_vd_penalty(t)
    expected = torch.from_numpy(_reference(t.numpy()))
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=0)


def test_vd_penalty_matches_reference_series_branch():
    t = torch.tensor([-5.1, -8.0, -12.0], dtype=torch.float64)
    out = _log_uniform_vd_penalty(t)
    expected = torch.from_numpy(_reference(t.numpy()))
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=0)


def test_vd_penalty_seam_continuity():
    t = torch.tensor([-5.0 - 1e-4, -5.0 + 1e-4], dtype=torch.float64)
    lo, hi = _log_uniform_vd_penalty(t)
    assert abs(lo - hi) / hi < 1e-3


def test_vd_penalty_extreme_regime_finite_and_tiny():
    """Regression: a fully-pruned weight (huge log_alpha) used to give an
    inf penalty and NaN gradients through expi(-0.0)."""
    layer = LinearVD(4, 6)
    with torch.no_grad():
        layer.log_sigma2.fill_(300.0)
    p = layer.penalty
    assert torch.isfinite(p).all()
    assert (p >= 0).all()
    assert p.max().item() < 1e-30
    p.sum().backward()
    assert torch.isfinite(layer.log_sigma2.grad).all()
    assert torch.isfinite(layer.weight.grad.real).all()
    assert torch.isfinite(layer.weight.grad.imag).all()


def test_vd_penalty_gradcheck_series_branch():
    t = torch.tensor([-30.0, -12.0, -6.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_log_uniform_vd_penalty, (t,), eps=1e-6, atol=1e-4)


def test_vd_penalty_gradcheck_exact_branch():
    t = torch.tensor([-4.0, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_log_uniform_vd_penalty, (t,), eps=1e-6, atol=1e-4)
