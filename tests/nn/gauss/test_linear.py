"""Tests for the Gauss-trick complex Linear layer."""

from __future__ import annotations

import pytest
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


def test_gauss_linear_weight_setter_fans_out():
    layer = Linear(8, 4, bias=True)
    target = torch.randn_like(layer.weight)
    layer.weight = target
    assert torch.equal(layer.linear_r.weight, target.real)
    assert torch.equal(layer.linear_i.weight, target.imag)
    assert torch.equal(layer.weight, target)


def test_gauss_linear_bias_setter_fans_out():
    layer = Linear(8, 4, bias=True)
    target = torch.randn(4, dtype=torch.cfloat)
    layer.bias = target
    assert torch.equal(layer.bias_r, target.real)
    assert torch.equal(layer.bias_i, target.imag)
    assert torch.equal(layer.bias, target)


def test_gauss_linear_bias_setter_raises_when_bias_disabled():
    layer = Linear(8, 4, bias=False)
    with pytest.raises(RuntimeError, match="bias=False"):
        layer.bias = torch.randn(4, dtype=torch.cfloat)


def test_gauss_linear_weight_property_returns_fresh_storage():
    """Documents the silent-no-op: mutating the property's return value does
    NOT write through to the underlying real parameters."""
    layer = Linear(8, 4)
    before_r = layer.linear_r.weight.detach().clone()
    before_i = layer.linear_i.weight.detach().clone()
    layer.weight.data.zero_()
    assert torch.equal(layer.linear_r.weight, before_r)
    assert torch.equal(layer.linear_i.weight, before_i)
