"""Tests for the differentiable Ei (exponential integral) wrapper."""

from __future__ import annotations

import scipy.special
import torch

from complextorch.nn.relevance._expi import torch_expi


def test_torch_expi_forward_matches_scipy():
    x = torch.tensor([-5.0, -1.0, 0.1, 1.0, 5.0])
    out = torch_expi(x).numpy()
    expected = scipy.special.expi(x.numpy())
    assert (abs(out - expected) < 1e-5).all()


def test_torch_expi_gradcheck():
    x = torch.tensor([0.5, 1.5, -2.0], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(torch_expi, (x,), eps=1e-6, atol=1e-4)


def test_torch_expi_forward_is_neg_inf_at_zero():
    """Ei(0) = -inf is the correct mathematical limit — not 'fixed'."""
    out = torch_expi(torch.tensor([0.0, -0.0]))
    assert (out == -torch.inf).all()


def test_torch_expi_backward_finite_at_zero():
    """Regression: e^x / x used to produce inf/NaN gradients at x = 0."""
    x = torch.tensor([0.0, -0.0, 1e-40, -1e-40], requires_grad=True)
    out = torch_expi(x)
    grad = torch.autograd.grad(out.sum(), x)[0]
    assert torch.isfinite(grad).all()
    # Sign follows the input side (copysign handles -0.0).
    assert (grad[torch.tensor([0, 2])] > 0).all()
    assert (grad[torch.tensor([1, 3])] < 0).all()
