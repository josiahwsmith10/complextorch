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
