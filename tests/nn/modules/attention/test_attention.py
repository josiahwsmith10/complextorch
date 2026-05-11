"""Tests for ScaledDotProductAttention and MultiheadAttention."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.attention import (
    MultiheadAttention,
    ScaledDotProductAttention,
)


def test_scaled_dot_product_complex_softmax():
    attn = ScaledDotProductAttention(temperature=2.0, softmax_on="complex")
    q = torch.randn(2, 3, 5, 4, dtype=torch.cfloat)
    k = torch.randn(2, 3, 7, 4, dtype=torch.cfloat)
    v = torch.randn(2, 3, 7, 4, dtype=torch.cfloat)
    out = attn(q, k, v)
    assert out.shape == (2, 3, 5, 4)
    assert out.is_complex()


def test_scaled_dot_product_real_softmax():
    attn = ScaledDotProductAttention(temperature=2.0, softmax_on="real")
    q = torch.randn(2, 3, 5, 4, dtype=torch.cfloat)
    k = torch.randn(2, 3, 7, 4, dtype=torch.cfloat)
    v = torch.randn(2, 3, 7, 4, dtype=torch.cfloat)
    out = attn(q, k, v)
    assert out.shape == (2, 3, 5, 4)
    assert out.is_complex()


def test_scaled_dot_product_invalid_softmax_on():
    with pytest.raises(ValueError, match="softmax_on must be"):
        ScaledDotProductAttention(temperature=1.0, softmax_on="bogus")


def test_multihead_attention_forward():
    mha = MultiheadAttention(n_heads=2, d_model=8, d_k=4, d_v=4)
    q = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = mha(q, q, q)
    assert out.shape == q.shape
    assert out.is_complex()


def test_multihead_attention_cross():
    mha = MultiheadAttention(n_heads=2, d_model=8, d_k=4, d_v=4)
    q = torch.randn(2, 5, 8, dtype=torch.cfloat)
    k = torch.randn(2, 7, 8, dtype=torch.cfloat)
    v = torch.randn(2, 7, 8, dtype=torch.cfloat)
    out = mha(q, k, v)
    assert out.shape == q.shape
