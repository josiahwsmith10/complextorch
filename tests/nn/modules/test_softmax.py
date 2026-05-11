"""Tests for complex softmax variants."""

from __future__ import annotations

import torch

from complextorch.nn.modules.softmax import CVSoftMax, MagSoftMax, PhaseSoftMax


def test_cvsoftmax():
    sm = CVSoftMax(dim=-1)
    x = torch.randn(3, 5, dtype=torch.cfloat)
    out = sm(x)
    assert out.shape == x.shape
    # Real part sums to 1 along dim
    torch.testing.assert_close(
        out.real.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        out.imag.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5
    )


def test_magsoftmax_returns_real():
    sm = MagSoftMax(dim=-1)
    x = torch.randn(3, 5, dtype=torch.cfloat)
    out = sm(x)
    assert not out.is_complex()
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)


def test_phasesoftmax_preserves_phase():
    sm = PhaseSoftMax(dim=-1)
    x = torch.randn(3, 5, dtype=torch.cfloat) + 0.5  # avoid |z|=0
    out = sm(x)
    assert out.is_complex()
    torch.testing.assert_close(out.angle(), x.angle(), atol=1e-5, rtol=1e-5)
