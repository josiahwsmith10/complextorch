"""Tests for complex positional encodings (RoPE / sinusoidal / CoPE)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from complextorch.nn.modules.position import (
    CoPE,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
)

# ---------- RotaryEmbedding ----------


def test_rotary_shape_and_unit_modulus():
    rope = RotaryEmbedding(dim=8)
    x = torch.randn(2, 4, 6, 8, dtype=torch.cfloat)
    out = rope(x)
    assert out.shape == x.shape
    assert out.is_complex()
    # rotation preserves magnitude
    torch.testing.assert_close(out.abs(), x.abs(), atol=1e-5, rtol=1e-5)


def test_rotary_relative_position_property():
    """<RoPE(q, m), RoPE(k, n)> depends only on the offset (m - n)."""
    dim, length = 8, 6
    rope = RotaryEmbedding(dim)
    q = torch.randn(1, dim, dtype=torch.cfloat)
    k = torch.randn(1, dim, dtype=torch.cfloat)
    qs = rope(q.expand(length, dim))  # same feature at every position
    ks = rope(k.expand(length, dim))
    scores = qs @ ks.conj().transpose(-2, -1)  # (L, L) Hermitian inner products
    # Each diagonal corresponds to a fixed (m - n) and must be constant.
    for offset in range(-(length - 1), length):
        diag = torch.diagonal(scores, offset=offset)
        torch.testing.assert_close(diag, diag[0].expand_as(diag), atol=1e-5, rtol=1e-5)


def test_rotary_offset_semantics():
    dim = 4
    rope = RotaryEmbedding(dim)
    x = torch.ones(3, dim, dtype=torch.cfloat)
    out0 = rope(x, offset=0)
    out2 = rope(x, offset=2)
    assert not torch.allclose(out0, out2)
    full = rope(torch.ones(5, dim, dtype=torch.cfloat))
    torch.testing.assert_close(out2[0], full[2], atol=1e-5, rtol=1e-5)


def test_rotary_real_input_auto_casts():
    out = RotaryEmbedding(4)(torch.randn(3, 4))
    assert out.is_complex()


def test_rotary_dim_mismatch_raises():
    with pytest.raises(ValueError, match="must equal dim"):
        RotaryEmbedding(4)(torch.randn(3, 5, dtype=torch.cfloat))


def test_rotary_learnable_grad_flows():
    rope = RotaryEmbedding(4, learnable=True)
    assert isinstance(rope.inv_freq, nn.Parameter)
    x = torch.randn(3, 4, dtype=torch.cfloat)
    # phase-dependent loss so the gradient w.r.t. frequencies is non-trivial
    rope(x).real.sum().backward()
    assert rope.inv_freq.grad is not None
    assert torch.isfinite(rope.inv_freq.grad).all()


def test_rotary_rotate_q_k():
    rope = RotaryEmbedding(4)
    q = torch.randn(2, 3, 4, dtype=torch.cfloat)
    k = torch.randn(2, 5, 4, dtype=torch.cfloat)
    rq, rk = rope.rotate_q_k(q, k)
    assert rq.shape == q.shape
    assert rk.shape == k.shape


def test_rotary_extra_repr():
    s = RotaryEmbedding(4, base=100.0, learnable=True).extra_repr()
    assert "dim=4" in s
    assert "learnable=True" in s


# ---------- SinusoidalPositionalEncoding ----------


def test_sinusoidal_shape_and_additive():
    pe = SinusoidalPositionalEncoding(dim=6)
    x = torch.randn(2, 5, 6, dtype=torch.cfloat)
    out = pe(x)
    assert out.shape == x.shape
    assert out.is_complex()
    # difference equals the (input-independent) positional bank
    delta = out - x
    delta2 = pe(torch.zeros_like(x))
    torch.testing.assert_close(delta, delta2, atol=1e-5, rtol=1e-5)


def test_sinusoidal_real_input_auto_casts():
    out = SinusoidalPositionalEncoding(4)(torch.randn(3, 4))
    assert out.is_complex()


def test_sinusoidal_dim_mismatch_raises():
    with pytest.raises(ValueError, match="must equal dim"):
        SinusoidalPositionalEncoding(4)(torch.randn(3, 5, dtype=torch.cfloat))


def test_sinusoidal_extra_repr():
    assert "dim=4" in SinusoidalPositionalEncoding(4).extra_repr()


# ---------- CoPE ----------


def test_cope_shape_and_unit_modulus():
    cope = CoPE(dim=8)
    x = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = cope(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.abs(), x.abs(), atol=1e-5, rtol=1e-5)


def test_cope_grad_flows_into_both_params():
    cope = CoPE(4)
    x = torch.randn(3, 4, dtype=torch.cfloat)
    cope(x).real.sum().backward()
    assert cope.omega.grad is not None
    assert cope.phi.grad is not None
    assert torch.isfinite(cope.omega.grad).all()


def test_cope_real_input_auto_casts():
    out = CoPE(4)(torch.randn(3, 4))
    assert out.is_complex()


def test_cope_dim_mismatch_raises():
    with pytest.raises(ValueError, match="must equal dim"):
        CoPE(4)(torch.randn(3, 5, dtype=torch.cfloat))


def test_cope_extra_repr():
    assert "dim=4" in CoPE(4).extra_repr()
