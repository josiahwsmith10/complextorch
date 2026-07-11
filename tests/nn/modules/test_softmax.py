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


def test_cvsoftmax_mask_zeroes_masked_positions():
    """-inf mask on both parts -> masked entries exactly 0+0j, kept parts
    renormalize to 1 in each split softmax."""
    sm = CVSoftMax(dim=-1)
    z = torch.randn(2, 5, dtype=torch.cfloat)
    mask = torch.zeros(2, 5)
    mask[:, -2:] = float("-inf")
    out = sm(z, mask)
    assert torch.equal(out[:, -2:], torch.zeros(2, 2, dtype=torch.cfloat))
    torch.testing.assert_close(out[:, :3].real.sum(-1), torch.ones(2))
    torch.testing.assert_close(out[:, :3].imag.sum(-1), torch.ones(2))


def test_phasesoftmax_mask_zeroes_and_preserves_phase():
    sm = PhaseSoftMax(dim=-1)
    z = torch.randn(2, 5, dtype=torch.cfloat) + 0.1
    mask = torch.zeros(2, 5)
    mask[:, 0] = float("-inf")
    out = sm(z, mask)
    assert torch.equal(out[:, 0], torch.zeros(2, dtype=torch.cfloat))
    torch.testing.assert_close(out[:, 1:].abs().sum(-1), torch.ones(2))
    torch.testing.assert_close(out[:, 1:].angle(), z[:, 1:].angle())


def test_magsoftmax_mask_zeroes_masked_positions():
    sm = MagSoftMax(dim=-1)
    z = torch.randn(2, 5, dtype=torch.cfloat)
    mask = torch.zeros(2, 5)
    mask[:, 2] = float("-inf")
    out = sm(z, mask)
    assert torch.equal(out[:, 2], torch.zeros(2))
    torch.testing.assert_close(out.sum(-1), torch.ones(2))
