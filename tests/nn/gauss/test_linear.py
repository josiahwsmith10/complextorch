"""Tests for the Gauss-trick complex Linear layer."""

from __future__ import annotations

import torch

from complextorch.nn.gauss.linear import Linear


def test_gauss_linear_forward():
    layer = Linear(8, 4, bias=True)
    x = torch.randn(3, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (3, 4)
    assert out.is_complex()
    # weight property
    w = layer.weight
    assert w.shape == (4, 8)
    assert w.is_complex()
    # bias property
    b = layer.bias
    assert b.shape == (4,)


def test_gauss_linear_no_bias():
    layer = Linear(8, 4, bias=False)
    x = torch.randn(3, 8, dtype=torch.cfloat)
    out = layer(x)
    assert out.shape == (3, 4)
    assert layer.bias is None
