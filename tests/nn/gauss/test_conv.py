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


@pytest.mark.parametrize(
    ("cls", "kwargs", "wr_attr", "wi_attr"),
    [
        (
            Conv1d,
            {"in_channels": 2, "out_channels": 4, "kernel_size": 3},
            "conv_r",
            "conv_i",
        ),
        (
            Conv2d,
            {"in_channels": 2, "out_channels": 3, "kernel_size": 3},
            "conv_r",
            "conv_i",
        ),
        (
            Conv3d,
            {"in_channels": 2, "out_channels": 2, "kernel_size": 3},
            "conv_r",
            "conv_i",
        ),
        (
            ConvTranspose1d,
            {"in_channels": 2, "out_channels": 3, "kernel_size": 3},
            "convt_r",
            "convt_i",
        ),
        (
            ConvTranspose2d,
            {"in_channels": 2, "out_channels": 2, "kernel_size": 3},
            "convt_r",
            "convt_i",
        ),
        (
            ConvTranspose3d,
            {"in_channels": 2, "out_channels": 2, "kernel_size": 3},
            "convt_r",
            "convt_i",
        ),
    ],
)
def test_gauss_weight_setter_fans_out(cls, kwargs, wr_attr, wi_attr):
    layer = cls(**kwargs)
    target = torch.randn_like(layer.weight)
    layer.weight = target
    assert torch.equal(getattr(layer, wr_attr).weight, target.real)
    assert torch.equal(getattr(layer, wi_attr).weight, target.imag)
    # Reading back through the property reflects the new value.
    assert torch.equal(layer.weight, target)


@pytest.mark.parametrize(
    "cls",
    [Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d],
)
def test_gauss_bias_setter_fans_out(cls):
    layer = cls(in_channels=2, out_channels=3, kernel_size=3, bias=True)
    target = torch.randn(3, dtype=torch.cfloat)
    layer.bias = target
    assert torch.equal(layer.bias_r, target.real)
    assert torch.equal(layer.bias_i, target.imag)
    assert torch.equal(layer.bias, target)


@pytest.mark.parametrize(
    "cls",
    [Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d],
)
def test_gauss_bias_setter_raises_when_bias_disabled(cls):
    layer = cls(in_channels=2, out_channels=3, kernel_size=3, bias=False)
    with pytest.raises(RuntimeError, match="bias=False"):
        layer.bias = torch.randn(3, dtype=torch.cfloat)


def test_gauss_weight_property_returns_fresh_storage():
    """The @property accessor returns a new tensor — mutating it does NOT
    write through to the underlying split parameters. This documents the
    silent-no-op caveat that motivated adding the setter.
    """
    layer = Conv1d(in_channels=2, out_channels=3, kernel_size=3)
    before_r = layer.conv_r.weight.detach().clone()
    before_i = layer.conv_i.weight.detach().clone()
    layer.weight.data.zero_()
    assert torch.equal(layer.conv_r.weight, before_r)
    assert torch.equal(layer.conv_i.weight, before_i)


def test_gauss_weight_assignment_changes_forward_output():
    """End-to-end: setting weight via the setter should change the forward."""
    layer = Conv1d(in_channels=2, out_channels=3, kernel_size=3, bias=False)
    x = torch.randn(1, 2, 5, dtype=torch.cfloat)
    y_before = layer(x).detach().clone()
    layer.weight = torch.randn_like(layer.weight)
    y_after = layer(x)
    assert not torch.allclose(y_before, y_after)
