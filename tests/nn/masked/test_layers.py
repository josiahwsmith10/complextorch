"""Tests for the masked complex layers."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.masked import (
    BilinearMasked,
    Conv1dMasked,
    Conv2dMasked,
    Conv3dMasked,
    LinearMasked,
)


# ---------- LinearMasked / BilinearMasked ----------


def test_linear_masked_dense_forward():
    layer = LinearMasked(4, 6, bias=True)
    x = torch.randn(2, 4, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (2, 6)
    assert out.is_complex()


def test_linear_masked_sparse_forward():
    layer = LinearMasked(4, 6, bias=True)
    layer.mask = torch.zeros(6, 4)  # zero out everything
    x = torch.randn(2, 4, dtype=torch.cfloat)
    out = layer(x)
    # With zero mask, output equals bias broadcast.
    torch.testing.assert_close(out, layer.bias.expand_as(out))


def test_linear_masked_no_bias():
    layer = LinearMasked(4, 6, bias=False)
    assert layer.bias is None
    x = torch.randn(2, 4, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()


def test_bilinear_masked_forward():
    layer = BilinearMasked(4, 5, 6, bias=True)
    x1 = torch.randn(2, 4, dtype=torch.cfloat)
    x2 = torch.randn(2, 5, dtype=torch.cfloat)
    out = layer(x1, x2)
    assert out.shape == (2, 6)
    assert out.is_complex()


def test_bilinear_masked_no_conjugate():
    layer = BilinearMasked(4, 5, 6, conjugate=False)
    x1 = torch.randn(2, 4, dtype=torch.cfloat)
    x2 = torch.randn(2, 5, dtype=torch.cfloat)
    out = layer(x1, x2)
    assert out.shape == (2, 6)


def test_bilinear_masked_sparse_forward():
    layer = BilinearMasked(4, 5, 6, bias=True)
    layer.mask = torch.zeros(6, 4, 5)
    x1 = torch.randn(2, 4, dtype=torch.cfloat)
    x2 = torch.randn(2, 5, dtype=torch.cfloat)
    out = layer(x1, x2)
    torch.testing.assert_close(out, layer.bias.expand_as(out))


def test_bilinear_masked_no_bias():
    layer = BilinearMasked(4, 5, 6, bias=False)
    assert layer.bias is None


# ---------- ConvMasked ----------


@pytest.mark.parametrize(
    "cls, shape, k",
    [
        (Conv1dMasked, (1, 2, 8), 3),
        (Conv2dMasked, (1, 2, 6, 6), 3),
        (Conv3dMasked, (1, 2, 4, 4, 4), 3),
    ],
)
def test_conv_masked_dense_forward(cls, shape, k):
    layer = cls(in_channels=2, out_channels=4, kernel_size=k, padding=1)
    x = torch.randn(*shape, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()


def test_conv1d_masked_sparse():
    layer = Conv1dMasked(2, 4, kernel_size=3, padding=1)
    layer.mask = torch.zeros(4, 2, 3)
    x = torch.randn(1, 2, 8, dtype=torch.cfloat)
    out = layer(x)
    # Zero mask -> output is bias broadcast over output positions
    expected = layer.bias.view(1, 4, 1).expand_as(out)
    torch.testing.assert_close(out, expected)


def test_conv_masked_no_bias():
    layer = Conv1dMasked(2, 4, kernel_size=3, bias=False)
    assert layer.bias is None


def test_conv_masked_invalid_padding_mode():
    with pytest.raises(ValueError, match="padding_mode"):
        Conv1dMasked(2, 4, kernel_size=3, padding_mode="reflect")


def test_conv_masked_tuple_kernel_size():
    layer = Conv2dMasked(2, 4, kernel_size=(3, 5), padding=(1, 2))
    x = torch.randn(1, 2, 8, 10, dtype=torch.cfloat)
    out = layer(x)
    assert out.is_complex()


def test_conv_masked_str_padding():
    layer = Conv2dMasked(2, 4, kernel_size=3, padding="same")
    x = torch.randn(1, 2, 8, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (1, 4, 8, 8)
