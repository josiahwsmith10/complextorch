"""Tests for PhaseShift and ComplexScaling."""

from __future__ import annotations

import torch

from complextorch.nn.modules.phase import ComplexScaling, PhaseShift


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


# ---------- ComplexScaling ----------


def test_complex_scaling_shape_per_channel():
    layer = ComplexScaling(num_features=4)
    x = torch.randn(2, 4, 6, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_complex_scaling_scalar_broadcasts():
    layer = ComplexScaling(num_features=1)
    x = torch.randn(3, 5, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == x.shape


def test_complex_scaling_real_input_auto_casts():
    layer = ComplexScaling(num_features=4)
    x = torch.randn(2, 4, 6)
    out = layer(x)
    assert out.is_complex()


def test_complex_scaling_equivariant_under_phase_rotation():
    """ComplexScaling commutes with a global phase rotation."""
    layer = ComplexScaling(num_features=4)
    x = torch.randn(2, 4, 6, dtype=torch.cfloat)
    psi = torch.tensor(0.7)
    rotor = torch.polar(torch.tensor(1.0), psi)
    y1 = layer(x * rotor)
    y2 = layer(x) * rotor
    torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-5)


def test_complex_scaling_grad_flows():
    layer = ComplexScaling(num_features=3)
    x = torch.randn(2, 3, dtype=torch.cfloat, requires_grad=True)
    out = layer(x)
    out.abs().sum().backward()
    assert layer.alpha.grad is not None
    assert layer.beta.grad is not None
    assert torch.isfinite(layer.alpha.grad).all()


def test_complex_scaling_extra_repr():
    s = ComplexScaling(num_features=4).extra_repr()
    assert "num_features=4" in s
