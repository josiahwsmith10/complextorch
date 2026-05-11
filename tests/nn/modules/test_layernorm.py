"""Tests for complex LayerNorm."""

from __future__ import annotations

import torch

from complextorch.nn.modules.layernorm import LayerNorm


def test_layernorm_int_normalized_shape():
    ln = LayerNorm(8)
    x = torch.randn(4, 8, dtype=torch.cfloat)
    out = ln(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_layernorm_list_normalized_shape():
    ln = LayerNorm([4, 8])
    x = torch.randn(2, 3, 4, 8, dtype=torch.cfloat)
    out = ln(x)
    assert out.shape == x.shape


def test_layernorm_torch_size_normalized_shape():
    ln = LayerNorm(torch.Size([8]))
    x = torch.randn(2, 8, dtype=torch.cfloat)
    out = ln(x)
    assert out.shape == x.shape


def test_layernorm_no_affine():
    ln = LayerNorm(8, elementwise_affine=False)
    x = torch.randn(4, 8, dtype=torch.cfloat)
    out = ln(x)
    assert out.shape == x.shape
    assert ln.weight is None
    assert ln.bias is None


def test_layernorm_reset_parameters_no_op_no_affine():
    ln = LayerNorm(4, elementwise_affine=False)
    ln.reset_parameters()
