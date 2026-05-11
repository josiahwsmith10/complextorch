"""Tests for complex pooling layers."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.pooling import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MagMaxPool1d,
    MagMaxPool2d,
    MagMaxPool3d,
    SpectralPool1d,
    SpectralPool2d,
    SpectralPool3d,
)


@pytest.mark.parametrize(
    ("cls", "shape", "out_size"),
    [
        (AdaptiveAvgPool1d, (2, 4, 8), 4),
        (AdaptiveAvgPool2d, (2, 4, 8, 8), (4, 4)),
        (AdaptiveAvgPool3d, (1, 4, 4, 4, 4), (2, 2, 2)),
    ],
)
def test_adaptive_avg_pool(cls, shape, out_size):
    pool = cls(out_size)
    x = torch.randn(*shape, dtype=torch.cfloat)
    out = pool(x)
    assert out.is_complex()


@pytest.mark.parametrize(
    ("cls", "shape"),
    [
        (AvgPool1d, (2, 4, 8)),
        (AvgPool2d, (2, 4, 8, 8)),
        (AvgPool3d, (1, 4, 4, 4, 4)),
    ],
)
def test_avg_pool_complex(cls, shape):
    pool = cls(kernel_size=2)
    x = torch.randn(*shape, dtype=torch.cfloat)
    out = pool(x)
    assert out.is_complex()


@pytest.mark.parametrize(
    ("cls", "shape"),
    [
        (AvgPool1d, (2, 4, 8)),
        (AvgPool2d, (2, 4, 8, 8)),
        (AvgPool3d, (1, 4, 4, 4, 4)),
    ],
)
def test_avg_pool_real_passthrough(cls, shape):
    pool = cls(kernel_size=2)
    x = torch.randn(*shape)
    out = pool(x)
    assert not out.is_complex()


@pytest.mark.parametrize(
    ("cls", "shape"),
    [
        (MagMaxPool1d, (2, 4, 8)),
        (MagMaxPool2d, (2, 4, 8, 8)),
        (MagMaxPool3d, (1, 4, 4, 4, 4)),
    ],
)
def test_magmax_pool_complex(cls, shape):
    pool = cls(kernel_size=2)
    x = torch.randn(*shape, dtype=torch.cfloat)
    out = pool(x)
    assert out.is_complex()


def test_magmaxpool_real_input():
    pool = MagMaxPool1d(kernel_size=2)
    x = torch.randn(2, 4, 8)
    out = pool(x)
    assert not out.is_complex()


def test_magmaxpool_return_indices():
    pool = MagMaxPool2d(kernel_size=2, return_indices=True)
    x = torch.randn(1, 2, 4, 4, dtype=torch.cfloat)
    out, indices = pool(x)
    assert out.is_complex()
    assert indices.dtype == torch.int64


def test_magmaxpool_extra_repr():
    s = MagMaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1).extra_repr()
    assert "kernel_size=3" in s
    assert "stride=2" in s


# ---------------------------------------------------------------------------
# SpectralPool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "in_shape", "out_size", "expected_spatial"),
    [
        (SpectralPool1d, (2, 4, 16), 8, (8,)),
        (SpectralPool1d, (2, 4, 16), (8,), (8,)),
        (SpectralPool2d, (2, 4, 16, 16), 8, (8, 8)),
        (SpectralPool2d, (2, 4, 16, 12), (8, 6), (8, 6)),
        (SpectralPool3d, (1, 2, 8, 8, 8), 4, (4, 4, 4)),
        (SpectralPool3d, (1, 2, 8, 10, 12), (4, 6, 6), (4, 6, 6)),
    ],
)
def test_spectral_pool_shape(cls, in_shape, out_size, expected_spatial):
    pool = cls(out_size)
    x = torch.randn(*in_shape, dtype=torch.cfloat)
    out = pool(x)
    assert out.shape == in_shape[: -len(expected_spatial)] + expected_spatial
    assert out.is_complex()


@pytest.mark.parametrize(
    ("cls", "in_shape", "out_size"),
    [
        (SpectralPool1d, (2, 3, 16), 8),
        (SpectralPool2d, (2, 3, 16, 16), (8, 8)),
        (SpectralPool3d, (1, 2, 8, 8, 8), (4, 4, 4)),
    ],
)
def test_spectral_pool_real_input_returns_real(cls, in_shape, out_size):
    pool = cls(out_size)
    x = torch.randn(*in_shape)
    out = pool(x)
    assert not out.is_complex()
    assert out.shape[: -pool._ndim] == in_shape[: -pool._ndim]


@pytest.mark.parametrize(
    ("cls", "in_shape", "out_size"),
    [
        (SpectralPool1d, (2, 3, 16), 16),
        (SpectralPool2d, (2, 3, 8, 8), (8, 8)),
        (SpectralPool3d, (1, 2, 4, 4, 4), (4, 4, 4)),
    ],
)
def test_spectral_pool_identity_when_same_size(cls, in_shape, out_size):
    pool = cls(out_size)
    x = torch.randn(*in_shape, dtype=torch.cfloat)
    out = pool(x)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize(
    ("cls", "in_shape", "out_size"),
    [
        # cover even/odd N and K combinations to exercise the centered-crop math
        (SpectralPool1d, (2, 3, 16), 8),
        (SpectralPool1d, (2, 3, 17), 5),
        (SpectralPool1d, (2, 3, 16), 5),
        (SpectralPool1d, (2, 3, 17), 8),
        (SpectralPool2d, (2, 3, 12, 14), (6, 9)),
        (SpectralPool3d, (1, 2, 6, 7, 8), (3, 5, 4)),
    ],
)
def test_spectral_pool_preserves_spatial_mean(cls, in_shape, out_size):
    pool = cls(out_size)
    x = torch.randn(*in_shape, dtype=torch.cfloat)
    out = pool(x)
    spatial_dims = tuple(range(-pool._ndim, 0))
    torch.testing.assert_close(
        out.mean(dim=spatial_dims), x.mean(dim=spatial_dims), atol=1e-5, rtol=1e-5
    )


def test_spectral_pool_gradient_flows():
    pool = SpectralPool2d((4, 4))
    x = torch.randn(2, 3, 8, 8, dtype=torch.cfloat, requires_grad=True)
    out = pool(x)
    out.abs().pow(2).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_spectral_pool_low_pass_rejects_high_freq_tone():
    # Pure high-frequency tone should be (mostly) zeroed by low-pass pooling.
    N, K = 32, 8
    n = torch.arange(N, dtype=torch.float32)
    # frequency at the Nyquist (highest possible bin) — definitely outside the kept window
    tone = torch.exp(1j * torch.pi * n).to(torch.cfloat)
    x = tone.reshape(1, 1, N)
    pool = SpectralPool1d(K)
    out = pool(x)
    # The kept window is symmetric around DC and only spans the lowest K bins;
    # a Nyquist-frequency tone has all its energy outside that window.
    assert out.abs().max().item() < 1e-5


def test_spectral_pool_low_pass_keeps_dc():
    # A DC signal should pass through unchanged in mean.
    x = torch.full((1, 1, 16), 3.0 + 2.0j, dtype=torch.cfloat)
    pool = SpectralPool1d(4)
    out = pool(x)
    torch.testing.assert_close(
        out,
        torch.full((1, 1, 4), 3.0 + 2.0j, dtype=torch.cfloat),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("cls", "in_shape", "bad_size"),
    [
        (SpectralPool1d, (1, 1, 8), 16),
        (SpectralPool2d, (1, 1, 8, 8), (16, 8)),
        (SpectralPool3d, (1, 1, 4, 4, 4), (5, 4, 4)),
    ],
)
def test_spectral_pool_rejects_oversized_output(cls, in_shape, bad_size):
    pool = cls(bad_size)
    x = torch.randn(*in_shape, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="must not exceed"):
        pool(x)


def test_spectral_pool_rejects_nonpositive_output():
    pool = SpectralPool1d(0)
    x = torch.randn(1, 1, 8, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="must be positive"):
        pool(x)


def test_spectral_pool_rejects_bad_tuple_length():
    with pytest.raises(ValueError, match="must have length 2"):
        SpectralPool2d((4, 4, 4))


def test_spectral_pool_extra_repr():
    s = SpectralPool2d((4, 6)).extra_repr()
    assert "output_size=(4, 6)" in s
