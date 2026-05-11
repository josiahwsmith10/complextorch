"""Tests for wFM modules: convolutions, ReLU, distance linear."""

from __future__ import annotations

import math

import torch

from complextorch.nn.modules.manifold import (
    tReLU,
    wFMConv1d,
    wFMConv2d,
    wFMConvStrict2d,
    wFMDistanceLinear,
    wFMReLU,
)


def test_wfm_conv2d_forward():
    conv = wFMConv2d(in_channels=3, out_channels=5, kernel_size=(3, 3), padding=(1, 1))
    x = torch.randn(2, 3, 8, 8, dtype=torch.cfloat) + 0.1
    out = conv(x)
    assert out.is_complex()
    assert out.shape[0] == 2


def test_wfm_conv2d_fold_cache_reused():
    conv = wFMConv2d(in_channels=3, out_channels=5, kernel_size=(3, 3), padding=(1, 1))
    x = torch.randn(2, 3, 8, 8, dtype=torch.cfloat) + 0.1
    conv(x)
    n_before = len(conv._fold_cache)
    conv(x)
    n_after = len(conv._fold_cache)
    assert n_before == n_after


def test_wfm_conv1d_forward():
    conv = wFMConv1d(in_channels=3, out_channels=5, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8, dtype=torch.cfloat) + 0.1
    out = conv(x)
    assert out.is_complex()


def test_wfm_conv1d_weight_properties():
    conv = wFMConv1d(in_channels=3, out_channels=5, kernel_size=3)
    assert conv.weight_matrix_ang is conv.conv1d.weight_matrix_ang
    assert conv.weight_matrix_mag is conv.conv1d.weight_matrix_mag


# ---------- wFMReLU ----------


def test_wfm_relu_forward_shape():
    layer = wFMReLU(num_channels=4)
    x = torch.randn(2, 4, 6, 6, dtype=torch.cfloat) + 0.1
    out = layer(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_wfm_relu_grad_flows():
    layer = wFMReLU(num_channels=3)
    x = torch.randn(2, 3, 4, dtype=torch.cfloat) + 0.1
    out = layer(x)
    out.abs().sum().backward()
    assert layer.weight_phase.grad is not None
    assert layer.weight_mag.grad is not None
    assert torch.isfinite(layer.weight_phase.grad).all()


def test_wfm_relu_extra_repr():
    s = wFMReLU(num_channels=4).extra_repr()
    assert "num_channels=4" in s


# ---------- wFMDistanceLinear ----------


def test_wfm_distance_linear_returns_real():
    layer = wFMDistanceLinear(input_dim=4 * 6)
    x = torch.randn(2, 4, 6, dtype=torch.cfloat) + 0.1
    out = layer(x)
    assert out.shape == x.shape  # preserves shape of complex input
    assert not out.is_complex()
    assert out.dtype == torch.float32 or out.dtype == torch.float64


def test_wfm_distance_linear_grad_flows():
    layer = wFMDistanceLinear(input_dim=12)
    x = torch.randn(2, 3, 4, dtype=torch.cfloat) + 0.1
    out = layer(x)
    out.sum().backward()
    assert layer.weights.grad is not None
    assert torch.isfinite(layer.weights.grad).all()


def test_wfm_distance_linear_extra_repr():
    s = wFMDistanceLinear(input_dim=10).extra_repr()
    assert "input_dim=10" in s


def test_wfm_relu_real_input_auto_casts():
    """Real input is auto-cast to cfloat (forward's else branch)."""
    layer = wFMReLU(num_channels=3)
    x = torch.randn(2, 3, 5) + 0.1
    out = layer(x)
    assert out.is_complex()


def test_wfm_distance_linear_real_input_auto_casts():
    """wFMDistanceLinear also auto-casts real input."""
    layer = wFMDistanceLinear(input_dim=12)
    x = torch.randn(2, 3, 4) + 0.1
    out = layer(x)
    assert torch.isfinite(out).all()


def test_wfm_distance_linear_wrong_input_dim_raises():
    """flat.shape[1] != input_dim triggers a ValueError."""
    import pytest

    layer = wFMDistanceLinear(input_dim=10)
    x = torch.randn(2, 3, 4, dtype=torch.cfloat) + 0.1  # 12 != 10
    with pytest.raises(ValueError, match="expects flattened input of size"):
        layer(x)


# ---------- tReLU (SurReal Eq. 21-22) ----------


def test_trelu_magnitude_clamps_below_one_to_one():
    """|z| < 1 → out has magnitude exactly 1 (rectified)."""
    z = torch.tensor([0.5 + 0j, 0.3 + 0.4j], dtype=torch.cfloat)  # both |z| < 1
    out = tReLU()(z)
    torch.testing.assert_close(out.abs(), torch.ones(2), atol=1e-6, rtol=0)


def test_trelu_magnitude_preserves_above_one():
    """|z| >= 1 → magnitude unchanged."""
    z = torch.tensor([2.0 + 0j, 3.0 + 4.0j], dtype=torch.cfloat)  # |z| = 2 and 5
    out = tReLU()(z)
    torch.testing.assert_close(out.abs(), torch.tensor([2.0, 5.0]), atol=1e-5, rtol=0)


def test_trelu_phase_clamps_negative_to_zero():
    """arg(z) < 0 → output phase is exactly 0."""
    # arg = -π/4, |z| = 2 (above 1, so magnitude preserved)
    z = torch.tensor([math.sqrt(2.0) - 1j * math.sqrt(2.0)], dtype=torch.cfloat)
    out = tReLU()(z)
    torch.testing.assert_close(out.angle(), torch.zeros(1), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.abs(), torch.tensor([2.0]), atol=1e-5, rtol=0)


def test_trelu_phase_preserves_positive():
    """arg(z) >= 0 → phase unchanged."""
    z = torch.tensor([2.0 * (1 + 1j) / math.sqrt(2.0)], dtype=torch.cfloat)  # |z|=2, θ=π/4
    out = tReLU()(z)
    torch.testing.assert_close(out.angle(), torch.tensor([math.pi / 4]), atol=1e-5, rtol=0)


def test_trelu_no_parameters():
    """tReLU is parameter-free."""
    assert sum(p.numel() for p in tReLU().parameters()) == 0


def test_trelu_real_input_auto_casts():
    out = tReLU()(torch.tensor([0.5]))
    assert out.is_complex()


def test_trelu_grad_flows():
    """Gradient should flow through tReLU on the non-clipped regions."""
    z = (torch.randn(2, 3, dtype=torch.cfloat) + 1.5).requires_grad_(True)
    out = tReLU()(z)
    out.abs().sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


# ---------- wFMConvStrict2d (SurReal Eq. 14-16) ----------


def test_wfm_conv_strict_forward_shape():
    conv = wFMConvStrict2d(in_channels=3, out_channels=5, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8, 8, dtype=torch.cfloat) + 0.1
    out = conv(x)
    assert out.shape == (2, 5, 8, 8)
    assert out.is_complex()


def test_wfm_conv_strict_accepts_tuple_kernel_size():
    conv = wFMConvStrict2d(
        in_channels=2, out_channels=4, kernel_size=(3, 5), padding=(1, 2)
    )
    x = torch.randn(1, 2, 8, 8, dtype=torch.cfloat) + 0.1
    out = conv(x)
    assert out.shape == (1, 4, 8, 8)


def test_wfm_conv_strict_stride_reduces_spatial():
    conv = wFMConvStrict2d(in_channels=2, out_channels=4, kernel_size=3, stride=2)
    x = torch.randn(1, 2, 8, 8, dtype=torch.cfloat) + 0.1
    out = conv(x)
    # (8 + 0 - 3) // 2 + 1 = 3
    assert out.shape == (1, 4, 3, 3)


def test_wfm_conv_strict_weights_are_convex():
    """The convex weight projection sums to 1 and is non-negative per output channel."""
    conv = wFMConvStrict2d(in_channels=3, out_channels=4, kernel_size=3)
    w = conv._convex_weights()
    assert w.shape == (4, 3 * 3 * 3)
    torch.testing.assert_close(
        w.sum(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5
    )
    assert (w >= 0).all()


def test_wfm_conv_strict_is_u1_equivariant():
    """Strict wFM-Conv with convex weights is exactly U(1)-equivariant.

    Uses ``padding=0`` because zero-padding cannot transform under phase
    rotation (``0 · e^{jψ} = 0``) and the ``(log|z|, arg z)`` representation
    of zero is ambiguous. See class docstring.
    """
    conv = wFMConvStrict2d(in_channels=2, out_channels=3, kernel_size=3, padding=0)
    # Phases well within (-π, π) to avoid branch wrapping under rotation.
    torch.manual_seed(7)
    x = 0.5 * torch.randn(2, 2, 6, 6, dtype=torch.cfloat) + 1.5
    rotor = torch.polar(torch.tensor(1.0), torch.tensor(0.3))
    y1 = conv(x * rotor)
    y2 = conv(x) * rotor
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)


def test_wfm_conv_strict_grad_flows():
    conv = wFMConvStrict2d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    x = (torch.randn(2, 2, 6, 6, dtype=torch.cfloat) + 0.5).requires_grad_(True)
    out = conv(x)
    out.abs().sum().backward()
    assert conv.weight.grad is not None
    assert torch.isfinite(conv.weight.grad).all()


def test_wfm_conv_strict_rejects_wrong_in_channels():
    import pytest

    conv = wFMConvStrict2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
    with pytest.raises(ValueError, match="expected in_channels=3"):
        conv(torch.randn(1, 5, 8, 8, dtype=torch.cfloat))


def test_wfm_conv_strict_rejects_wrong_dim():
    import pytest

    conv = wFMConvStrict2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
    with pytest.raises(ValueError, match="expects 4-D input"):
        conv(torch.randn(3, 8, 8, dtype=torch.cfloat))


def test_wfm_conv_strict_real_input_auto_casts():
    conv = wFMConvStrict2d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    x = torch.randn(1, 2, 6, 6) + 0.5
    out = conv(x)
    assert out.is_complex()


def test_wfm_conv_strict_extra_repr():
    s = wFMConvStrict2d(in_channels=3, out_channels=5, kernel_size=3).extra_repr()
    assert "in_channels=3" in s
    assert "out_channels=5" in s
