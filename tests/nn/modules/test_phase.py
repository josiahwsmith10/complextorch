"""Tests for PhaseShift."""

from __future__ import annotations


import torch

from complextorch.nn.modules.phase import PhaseShift


def test_phase_shift_scalar():
    layer = PhaseShift(num_features=1)
    x = torch.randn(4, 5, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == x.shape
    # |out| == |x|; phase rotated
    torch.testing.assert_close(out.abs(), x.abs(), atol=1e-5, rtol=1e-5)


def test_phase_shift_per_channel():
    layer = PhaseShift(num_features=4)
    x = torch.randn(2, 4, 6, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.abs(), x.abs(), atol=1e-5, rtol=1e-5)


def test_phase_shift_higher_rank_phi():
    layer = PhaseShift(num_features=(4, 6))
    x = torch.randn(2, 4, 6, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == x.shape


def test_phase_shift_broadcast_dim_neg():
    layer = PhaseShift(num_features=3, broadcast_dim=-1)
    x = torch.randn(2, 4, 3, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == x.shape


def test_phase_shift_real_input_auto_casts():
    layer = PhaseShift(num_features=1)
    x = torch.randn(4, 5)
    out = layer(x)
    assert out.is_complex()


def test_phase_shift_extra_repr():
    s = PhaseShift(num_features=4).extra_repr()
    assert "num_features=4" in s
