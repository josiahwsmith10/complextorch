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
