"""Tests for PhaseDivConv / PhaseConjConv phase-modulation layers."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.phase_modulation import (
    PhaseConjConv1d,
    PhaseConjConv2d,
    PhaseConjConv3d,
    PhaseDivConv1d,
    PhaseDivConv2d,
    PhaseDivConv3d,
    _align_to_conv_grid,
)


@pytest.mark.parametrize(
    ("cls", "shape", "kernel"),
    [
        (PhaseDivConv1d, (2, 4, 16), 3),
        (PhaseDivConv2d, (2, 4, 8, 8), 3),
        (PhaseDivConv3d, (2, 4, 4, 4, 4), 3),
        (PhaseConjConv1d, (2, 4, 16), 3),
        (PhaseConjConv2d, (2, 4, 8, 8), 3),
        (PhaseConjConv3d, (2, 4, 4, 4, 4), 3),
    ],
)
def test_phase_modulation_forward(cls, shape, kernel):
    layer = cls(in_channels=4, kernel_size=kernel, padding=kernel // 2)
    x = torch.randn(*shape, dtype=torch.cfloat) + 0.1
    out = layer(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_phase_div_conv_is_u1_invariant():
    """Global phase rotation cancels in numerator and denominator."""
    layer = PhaseDivConv2d(in_channels=3, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 5, 5, dtype=torch.cfloat) + 0.1
    rotor = torch.polar(torch.tensor(1.0), torch.tensor(0.7))
    y1 = layer(x * rotor)
    y2 = layer(x)
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)


def test_phase_conj_conv_is_u1_invariant():
    """PhaseConjConv with a C-linear inner conv is also U(1)-invariant.

    For complex-linear ``g``, ``g(e^{jψ} x) = e^{jψ} g(x)``, so
    ``(e^{jψ} x) · conj(e^{jψ} g(x)) = x · conj(g(x))``.
    """
    layer = PhaseConjConv2d(in_channels=3, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 5, 5, dtype=torch.cfloat) + 0.1
    rotor = torch.polar(torch.tensor(1.0), torch.tensor(0.7))
    y1 = layer(x * rotor)
    y2 = layer(x)
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)


def test_phase_modulation_use_one_filter_false():
    """When use_one_filter=False, inner conv has out_channels=in_channels."""
    layer = PhaseDivConv2d(
        in_channels=4, kernel_size=3, padding=1, use_one_filter=False
    )
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat) + 0.1
    out = layer(x)
    assert out.shape == x.shape


def test_phase_modulation_center_crop_no_padding():
    """When inner conv shrinks spatial dims, input is sampled at the
    receptive-field centers (for stride=1 this is the old center crop)."""
    layer = PhaseDivConv2d(in_channels=3, kernel_size=3, padding=0)
    x = torch.randn(2, 3, 8, 8, dtype=torch.cfloat) + 0.1
    out = layer(x)
    # No padding + k=3 → shrink by 2 on each spatial dim → (8-2, 8-2)
    assert out.shape == (2, 3, 6, 6)


def test_align_passthrough_when_shape_matches():
    layer = PhaseDivConv2d(in_channels=3, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8, 8)
    out = _align_to_conv_grid(x, layer.conv.conv, (8, 8))
    assert out is x


def test_phase_modulation_strided_pairing():
    """Regression: stride > 1 used to pair x with g(x) via a contiguous
    center crop instead of sampling at the receptive-field centers."""
    layer = PhaseConjConv1d(in_channels=1, kernel_size=1, stride=2)
    x = torch.randn(1, 1, 8, dtype=torch.cfloat) + 0.1
    out = layer(x)
    # k=1, s=2: g(x)[j] is computed from x[2j], so y[j] = x[2j]*conj(g(x)[j]).
    g_x = layer.conv(x)
    expected = x[..., 0::2] * g_x.conj()
    torch.testing.assert_close(out, expected)


def test_phase_modulation_stride_one_matches_center_crop():
    """For stride=1 shrinking convs the sampling equals the old center crop."""
    layer = PhaseDivConv1d(in_channels=2, kernel_size=5, padding=0)
    x = torch.randn(1, 2, 12, dtype=torch.cfloat) + 0.1
    out = layer(x)
    g_x = layer.conv(x).expand(1, 2, 8)
    denom = (g_x.real**2 + g_x.imag**2 + layer.eps).to(x.dtype)
    expected = x[..., 2:10] * g_x.conj() / denom
    torch.testing.assert_close(out, expected)


def test_phase_modulation_strided_2d_asymmetric():
    """2-D case with different spatial sizes per dim keeps per-dim geometry."""
    layer = PhaseConjConv2d(in_channels=2, kernel_size=3, stride=2, padding=1)
    x = torch.randn(1, 2, 10, 6, dtype=torch.cfloat) + 0.1
    out = layer(x)
    g_x = layer.conv(x).expand(1, 2, 5, 3)
    # k=3, p=1, s=2: receptive-field centers sit at input positions 0,2,4,...
    expected = x[..., 0::2, 0::2] * g_x.conj()
    torch.testing.assert_close(out, expected)


def test_align_rejects_oversized_conv_output():
    """Over-padded inner convs produce more outputs than pairable samples."""
    layer = PhaseConjConv1d(in_channels=1, kernel_size=1, padding=2)
    x = torch.randn(1, 1, 8, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="too large to pair"):
        layer(x)


def test_phase_modulation_grad_flows():
    layer = PhaseDivConv2d(in_channels=3, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 5, 5, dtype=torch.cfloat, requires_grad=True) + 0.1
    out = layer(x)
    out.abs().sum().backward()
    # Inner conv weight should have grad
    conv_weight = layer.conv.conv.weight
    assert conv_weight.grad is not None
    assert torch.isfinite(conv_weight.grad).all()


def test_phase_modulation_extra_repr():
    s = PhaseDivConv2d(in_channels=4, kernel_size=3, padding=1).extra_repr()
    assert "in_channels=4" in s


def test_phase_modulation_invalid_nd_raises():
    """Direct construction of the internal base with nd not in (1, 2, 3) is rejected."""
    from complextorch.nn.modules.phase_modulation import _PhaseDivConvNd

    with pytest.raises(ValueError, match="nd must be 1, 2, or 3"):
        _PhaseDivConvNd(nd=4, in_channels=2, kernel_size=3)


def test_phase_modulation_real_input_auto_casts():
    """Real input is upcast to cfloat in forward."""
    layer = PhaseDivConv1d(in_channels=3, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8) + 0.1
    out = layer(x)
    assert out.is_complex()
