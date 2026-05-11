"""Tests for the Gauss-trick complex Conv* and ConvTranspose* layers."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.gauss.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)


@pytest.mark.parametrize(
    ("cls", "x_shape", "kwargs"),
    [
        (
            Conv1d,
            (2, 2, 8),
            {"in_channels": 2, "out_channels": 4, "kernel_size": 3, "padding": 1},
        ),
        (
            Conv2d,
            (2, 2, 6, 6),
            {"in_channels": 2, "out_channels": 3, "kernel_size": 3, "padding": 1},
        ),
        (
            Conv3d,
            (1, 2, 4, 4, 4),
            {"in_channels": 2, "out_channels": 2, "kernel_size": 3, "padding": 1},
        ),
    ],
)
def test_gauss_conv_forward(cls, x_shape, kwargs):
    layer = cls(**kwargs)
    x = torch.randn(*x_shape, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()
    assert layer.weight.is_complex()
    assert layer.bias.is_complex()


@pytest.mark.parametrize(
    ("cls", "x_shape", "kwargs"),
    [
        (
            Conv1d,
            (2, 2, 8),
            {
                "in_channels": 2,
                "out_channels": 4,
                "kernel_size": 3,
                "padding": 1,
                "bias": False,
            },
        ),
        (
            Conv2d,
            (1, 2, 6, 6),
            {
                "in_channels": 2,
                "out_channels": 2,
                "kernel_size": 3,
                "padding": 1,
                "bias": False,
            },
        ),
    ],
)
def test_gauss_conv_no_bias(cls, x_shape, kwargs):
    layer = cls(**kwargs)
    x = torch.randn(*x_shape, dtype=torch.cfloat)
    out = layer(x)
    assert layer.bias is None
    assert out.is_complex()


@pytest.mark.parametrize(
    ("cls", "x_shape", "kwargs"),
    [
        (
            ConvTranspose1d,
            (2, 2, 4),
            {"in_channels": 2, "out_channels": 3, "kernel_size": 3, "stride": 2},
        ),
        (
            ConvTranspose2d,
            (1, 2, 4, 4),
            {"in_channels": 2, "out_channels": 2, "kernel_size": 3, "stride": 2},
        ),
        (
            ConvTranspose3d,
            (1, 2, 2, 2, 2),
            {"in_channels": 2, "out_channels": 2, "kernel_size": 3, "stride": 2},
        ),
    ],
)
def test_gauss_convtranspose_forward(cls, x_shape, kwargs):
    layer = cls(**kwargs)
    x = torch.randn(*x_shape, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()
    assert layer.weight.is_complex()
    assert layer.bias.is_complex()


@pytest.mark.parametrize(
    ("cls", "x_shape", "kwargs"),
    [
        (
            ConvTranspose1d,
            (2, 2, 4),
            {"in_channels": 2, "out_channels": 3, "kernel_size": 3, "bias": False},
        ),
        (
            ConvTranspose2d,
            (1, 2, 4, 4),
            {"in_channels": 2, "out_channels": 2, "kernel_size": 3, "bias": False},
        ),
    ],
)
def test_gauss_convtranspose_no_bias(cls, x_shape, kwargs):
    layer = cls(**kwargs)
    x = torch.randn(*x_shape, dtype=torch.cfloat)
    out = layer(x)
    assert layer.bias is None
    assert out.is_complex()
