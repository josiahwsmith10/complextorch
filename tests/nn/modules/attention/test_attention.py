"""Tests for ScaledDotProductAttention and MultiheadAttention."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.attention import (
    MultiheadAttention,
    ScaledDotProductAttention,
)
from complextorch.nn.modules.position import RotaryEmbedding


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


def test_multihead_attention_with_rotary():
    mha = MultiheadAttention(
        n_heads=2, d_model=8, d_k=4, d_v=4, rotary=RotaryEmbedding(4)
    )
    q = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = mha(q, q, q)
    assert out.shape == q.shape
    assert out.is_complex()


def test_multihead_attention_holographic():
    mha = MultiheadAttention(
        n_heads=2, d_model=8, d_k=4, d_v=4, attention="holographic"
    )
    q = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = mha(q, q, q)
    assert out.shape == q.shape
    assert out.is_complex()


def test_multihead_attention_invalid_attention():
    with pytest.raises(ValueError, match="attention must be"):
        MultiheadAttention(n_heads=2, d_model=8, d_k=4, d_v=4, attention="bogus")


# ---------- Masking ----------


def _reduced_key_reference(attn, q, k, v, n_keep):
    """Attention over only the first n_keep keys/values."""
    return attn(q, k[..., :n_keep, :], v[..., :n_keep, :])


@pytest.mark.parametrize("softmax_on", ["real", "complex"])
def test_sdpa_bool_mask_equals_truncated_keys(softmax_on):
    """Masking the trailing keys must equal attention on truncated k/v."""
    attn = ScaledDotProductAttention(
        temperature=2.0, attn_dropout=0.0, softmax_on=softmax_on
    )
    q = torch.randn(2, 4, 6, 4, dtype=torch.cfloat)
    k = torch.randn(2, 4, 7, 4, dtype=torch.cfloat)
    v = torch.randn(2, 4, 7, 4, dtype=torch.cfloat)
    mask = torch.zeros(6, 7, dtype=torch.bool)
    mask[:, 5:] = True  # disallow the last two keys
    out = attn(q, k, v, attn_mask=mask)
    expected = _reduced_key_reference(attn, q, k, v, 5)
    torch.testing.assert_close(out, expected)


def test_sdpa_phase_softmax_mask_equals_truncated_keys():
    from complextorch.nn.modules.softmax import PhaseSoftMax

    attn = ScaledDotProductAttention(
        temperature=2.0, attn_dropout=0.0, SoftMaxClass=PhaseSoftMax
    )
    q = torch.randn(1, 2, 5, 4, dtype=torch.cfloat)
    k = torch.randn(1, 2, 6, 4, dtype=torch.cfloat)
    v = torch.randn(1, 2, 6, 4, dtype=torch.cfloat)
    mask = torch.zeros(5, 6, dtype=torch.bool)
    mask[:, 4:] = True
    out = attn(q, k, v, attn_mask=mask)
    expected = _reduced_key_reference(attn, q, k, v, 4)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("softmax_on", ["real", "complex"])
def test_sdpa_float_mask_equals_bool_mask(softmax_on):
    attn = ScaledDotProductAttention(
        temperature=2.0, attn_dropout=0.0, softmax_on=softmax_on
    )
    q = torch.randn(1, 2, 4, 4, dtype=torch.cfloat)
    k = torch.randn(1, 2, 5, 4, dtype=torch.cfloat)
    v = torch.randn(1, 2, 5, 4, dtype=torch.cfloat)
    bool_mask = torch.zeros(4, 5, dtype=torch.bool)
    bool_mask[:, -1] = True
    float_mask = torch.zeros(4, 5)
    float_mask[:, -1] = float("-inf")
    torch.testing.assert_close(
        attn(q, k, v, attn_mask=bool_mask), attn(q, k, v, attn_mask=float_mask)
    )


def test_sdpa_integer_mask_dtype_raises():
    attn = ScaledDotProductAttention(temperature=2.0, attn_dropout=0.0)
    q = torch.randn(1, 2, 4, 4, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="attn_mask must be a bool or floating"):
        attn(q, q, q, attn_mask=torch.zeros(4, 4, dtype=torch.long))


def test_sdpa_masked_keys_get_zero_value_grad():
    """v.grad rows for fully masked-out keys must be exactly zero; all grads
    finite (the -inf logits do not leak NaN into backward)."""
    attn = ScaledDotProductAttention(temperature=2.0, attn_dropout=0.0)
    q = torch.randn(1, 1, 4, 4, dtype=torch.cfloat, requires_grad=True)
    k = torch.randn(1, 1, 5, 4, dtype=torch.cfloat, requires_grad=True)
    v = torch.randn(1, 1, 5, 4, dtype=torch.cfloat, requires_grad=True)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[:, -1] = True
    out = attn(q, k, v, attn_mask=mask)
    out.abs().sum().backward()
    for t in (q, k, v):
        assert torch.isfinite(t.grad.real).all()
        assert torch.isfinite(t.grad.imag).all()
    assert torch.equal(v.grad[..., -1, :], torch.zeros_like(v.grad[..., -1, :]))


def test_sdpa_fully_masked_row_yields_nan():
    """torch parity: softmax over an all--inf row is NaN, not special-cased."""
    attn = ScaledDotProductAttention(
        temperature=2.0, attn_dropout=0.0, softmax_on="real"
    )
    q = torch.randn(1, 1, 2, 4, dtype=torch.cfloat)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    mask[0, :] = True  # row 0 fully masked
    out = attn(q, q, q, attn_mask=mask)
    assert torch.isnan(out.real[..., 0, :]).all()


def test_mha_key_padding_mask_equals_truncated_keys():
    torch.manual_seed(0)
    mha = MultiheadAttention(2, 8, 4, 4, dropout=0.0)
    mha.eval()
    q = torch.randn(2, 5, 8, dtype=torch.cfloat)
    kv = torch.randn(2, 6, 8, dtype=torch.cfloat)
    kpm = torch.zeros(2, 6, dtype=torch.bool)
    kpm[:, 4:] = True  # last two keys are padding
    out = mha(q, kv, kv, key_padding_mask=kpm)
    expected = mha(q, kv[:, :4], kv[:, :4])
    torch.testing.assert_close(out, expected)


def test_mha_attn_mask_and_key_padding_mask_combine():
    torch.manual_seed(0)
    mha = MultiheadAttention(2, 8, 4, 4, dropout=0.0)
    mha.eval()
    q = torch.randn(1, 4, 8, dtype=torch.cfloat)
    kv = torch.randn(1, 5, 8, dtype=torch.cfloat)
    attn_mask = torch.zeros(4, 5, dtype=torch.bool)
    attn_mask[:, 4] = True
    kpm = torch.zeros(1, 5, dtype=torch.bool)
    kpm[:, 3] = True
    combined = mha(q, kv, kv, attn_mask=attn_mask, key_padding_mask=kpm)
    expected = mha(q, kv[:, :3], kv[:, :3])
    torch.testing.assert_close(combined, expected)


def test_mha_key_padding_mask_wrong_shape_raises():
    mha = MultiheadAttention(2, 8, 4, 4, dropout=0.0)
    q = torch.randn(2, 5, 8, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="key_padding_mask must have shape"):
        mha(q, q, q, key_padding_mask=torch.zeros(2, 9, dtype=torch.bool))


def test_mha_no_mask_unchanged():
    """Backward compat: mask-free calls behave exactly as before."""
    torch.manual_seed(0)
    mha = MultiheadAttention(2, 8, 4, 4, dropout=0.0)
    mha.eval()
    x = torch.randn(2, 5, 8, dtype=torch.cfloat)
    torch.testing.assert_close(mha(x, x, x), mha(x, x, x))


def test_mha_residual_norm_false_returns_raw_projected_output():
    """residual_norm=False skips the internal residual + LayerNorm."""
    torch.manual_seed(0)
    mha = MultiheadAttention(2, 8, 4, 4, dropout=0.0, residual_norm=False)
    mha.eval()
    assert mha.layer_norm is None
    x = torch.randn(2, 5, 8, dtype=torch.cfloat)
    out = mha(x, x, x)
    # Manual reference through the same submodules, no residual/norm tail.
    q = mha.w_q(x).view(2, 5, 2, 4).transpose(1, 2)
    k = mha.w_k(x).view(2, 5, 2, 4).transpose(1, 2)
    v = mha.w_v(x).view(2, 5, 2, 4).transpose(1, 2)
    a = mha.attention(q, k, v)
    expected = mha.fc(a.transpose(1, 2).contiguous().view(2, 5, -1))
    torch.testing.assert_close(out, expected)
