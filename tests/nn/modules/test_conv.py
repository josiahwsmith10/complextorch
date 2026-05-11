"""Tests for complex Conv* and ConvTranspose* layers (native cfloat wrappers)."""

from __future__ import annotations

import torch

from complextorch.nn.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
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
