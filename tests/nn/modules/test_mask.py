"""Tests for ComplexRatioMask / PhaseSigmoid / MagMinMaxNorm."""

from __future__ import annotations

import torch

from complextorch.nn.modules.mask import ComplexRatioMask, MagMinMaxNorm, PhaseSigmoid


def test_complex_ratio_mask_bounded_magnitude():
    m = ComplexRatioMask()
    x = torch.randn(8, dtype=torch.cfloat) * 10
    out = m(x)
    # sigmoid(|z|) <= 1
    assert out.abs().max().item() <= 1.0


def test_phase_sigmoid_bounded_magnitude():
    m = PhaseSigmoid()
    x = torch.randn(8, dtype=torch.cfloat) * 10
    out = m(x)
    assert out.abs().max().item() <= 1.0


def test_mag_min_max_norm_global():
    m = MagMinMaxNorm(dim=None)
    x = torch.tensor([1 + 0j, 2 + 0j, 5 + 0j], dtype=torch.cfloat)
    out = m(x)
    assert out.abs().min().item() == 0.0
    assert out.abs().max().item() == 1.0


def test_mag_min_max_norm_dim():
    m = MagMinMaxNorm(dim=-1)
    x = torch.tensor([[1 + 0j, 2 + 0j, 5 + 0j]], dtype=torch.cfloat)
    out = m(x)
    assert out.abs().min().item() == 0.0
    assert out.abs().max().item() == 1.0
