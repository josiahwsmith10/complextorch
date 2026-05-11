"""Tests for complex Conv* and SlowConv* layers."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    SlowConv1d,
    SlowConv2d,
    SlowConv3d,
    SlowConvTranspose1d,
    SlowConvTranspose2d,
    SlowConvTranspose3d,
)


# ---------- Fast variants ----------


def test_conv1d_forward_shape():
    layer = Conv1d(2, 4, kernel_size=3, padding=1)
    x = torch.randn(2, 2, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (2, 4, 8)
    assert out.is_complex()


def test_conv2d_forward_shape():
    layer = Conv2d(2, 4, kernel_size=3, padding=1)
    x = torch.randn(2, 2, 8, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (2, 4, 8, 8)


def test_conv3d_forward_shape():
    layer = Conv3d(2, 4, kernel_size=3, padding=1)
    x = torch.randn(1, 2, 4, 4, 4, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (1, 4, 4, 4, 4)


def test_convtranspose1d_forward_shape():
    layer = ConvTranspose1d(2, 4, kernel_size=3, stride=2)
    x = torch.randn(2, 2, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()


def test_convtranspose2d_forward_shape():
    layer = ConvTranspose2d(2, 4, kernel_size=3, stride=2)
    x = torch.randn(1, 2, 4, 4, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()


def test_convtranspose3d_forward_shape():
    layer = ConvTranspose3d(2, 4, kernel_size=3, stride=2)
    x = torch.randn(1, 2, 2, 2, 2, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()


# ---------- Slow variants: forward + weight/bias properties ----------


@pytest.mark.parametrize(
    "cls, x_shape, kwargs",
    [
        (
            SlowConv1d,
            (2, 2, 8),
            dict(in_channels=2, out_channels=4, kernel_size=3, padding=1),
        ),
        (
            SlowConv2d,
            (2, 2, 6, 6),
            dict(in_channels=2, out_channels=3, kernel_size=3, padding=1),
        ),
        (
            SlowConv3d,
            (1, 2, 4, 4, 4),
            dict(in_channels=2, out_channels=2, kernel_size=3, padding=1),
        ),
    ],
)
def test_slow_conv_forward(cls, x_shape, kwargs):
    layer = cls(**kwargs)
    x = torch.randn(*x_shape, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()
    # weight/bias properties
    assert layer.weight.is_complex()
    assert layer.bias.is_complex()


@pytest.mark.parametrize(
    "cls, x_shape, kwargs",
    [
        (
            SlowConv1d,
            (2, 2, 8),
            dict(in_channels=2, out_channels=4, kernel_size=3, padding=1, bias=False),
        ),
        (
            SlowConv2d,
            (1, 2, 6, 6),
            dict(in_channels=2, out_channels=2, kernel_size=3, padding=1, bias=False),
        ),
    ],
)
def test_slow_conv_no_bias(cls, x_shape, kwargs):
    layer = cls(**kwargs)
    x = torch.randn(*x_shape, dtype=torch.cfloat)
    out = layer(x)
    assert layer.bias is None
    assert out.is_complex()


@pytest.mark.parametrize(
    "cls, x_shape, kwargs",
    [
        (
            SlowConvTranspose1d,
            (2, 2, 4),
            dict(in_channels=2, out_channels=3, kernel_size=3, stride=2),
        ),
        (
            SlowConvTranspose2d,
            (1, 2, 4, 4),
            dict(in_channels=2, out_channels=2, kernel_size=3, stride=2),
        ),
        (
            SlowConvTranspose3d,
            (1, 2, 2, 2, 2),
            dict(in_channels=2, out_channels=2, kernel_size=3, stride=2),
        ),
    ],
)
def test_slow_convtranspose_forward(cls, x_shape, kwargs):
    layer = cls(**kwargs)
    x = torch.randn(*x_shape, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()
    assert layer.weight.is_complex()
    assert layer.bias.is_complex()


@pytest.mark.parametrize(
    "cls, x_shape, kwargs",
    [
        (
            SlowConvTranspose1d,
            (2, 2, 4),
            dict(in_channels=2, out_channels=3, kernel_size=3, bias=False),
        ),
        (
            SlowConvTranspose2d,
            (1, 2, 4, 4),
            dict(in_channels=2, out_channels=2, kernel_size=3, bias=False),
        ),
    ],
)
def test_slow_convtranspose_no_bias(cls, x_shape, kwargs):
    layer = cls(**kwargs)
    x = torch.randn(*x_shape, dtype=torch.cfloat)
    out = layer(x)
    assert layer.bias is None
    assert out.is_complex()
