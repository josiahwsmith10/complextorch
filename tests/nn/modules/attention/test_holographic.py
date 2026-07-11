"""Tests for HolographicAttention (interference-aware complex attention)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from complextorch.nn.modules.attention.holographic import HolographicAttention


@pytest.mark.parametrize("alpha_init", [0.25, 1.0, 3.0])
def test_holographic_effective_alpha_equals_alpha_init(alpha_init):
    """Regression: the raw parameter used to be alpha_init itself, making the
    effective weight softplus(alpha_init) instead of alpha_init."""
    attn = HolographicAttention(temperature=2.0, alpha_init=alpha_init)
    effective = F.softplus(attn.alpha)
    torch.testing.assert_close(
        effective, torch.tensor(alpha_init), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("alpha_init", [0.0, -1.0])
def test_holographic_nonpositive_alpha_init_raises(alpha_init):
    with pytest.raises(ValueError, match="alpha_init must be positive"):
        HolographicAttention(temperature=2.0, alpha_init=alpha_init)


def test_holographic_shape_and_dtype():
    attn = HolographicAttention(temperature=2.0)
    q = torch.randn(2, 3, 5, 4, dtype=torch.cfloat)
    k = torch.randn(2, 3, 7, 4, dtype=torch.cfloat)
    v = torch.randn(2, 3, 7, 4, dtype=torch.cfloat)
    out = attn(q, k, v)
    assert out.shape == (2, 3, 5, 4)
    assert out.is_complex()


def test_holographic_grad_flows_into_alpha():
    attn = HolographicAttention(temperature=2.0, attn_dropout=0.0)
    q = torch.randn(1, 4, 4, dtype=torch.cfloat)
    out = attn(q, q, q)
    (out.real**2 + out.imag**2).sum().backward()
    assert attn.alpha.grad is not None
    assert torch.isfinite(attn.alpha.grad).all()


def test_holographic_phase_gate_concentrates_on_aligned_value():
    """As alpha grows, the phase gate concentrates weight on the in-phase key."""
    d = 4
    qv = torch.randn(d, dtype=torch.cfloat)
    q = qv.unsqueeze(0)  # (1, d)
    theta = torch.tensor(0.9)
    rotated = qv * torch.polar(torch.ones(()), theta)
    k = torch.stack([qv, rotated])  # key 0 aligned, key 1 phase-rotated
    # Distinct values so the output reveals which key won.
    v = torch.stack(
        [torch.ones(d, dtype=torch.cfloat), -torch.ones(d, dtype=torch.cfloat)]
    )
    v0 = v[0]

    attn = HolographicAttention(temperature=1.0, attn_dropout=0.0)
    with torch.no_grad():
        attn.alpha.fill_(-5.0)  # softplus(-5) ~ 0 -> gate ~ 1 (phase-agnostic)
    out_weak = attn(q, k, v).squeeze(0)
    with torch.no_grad():
        attn.alpha.fill_(5.0)  # strong phase gate
    out_strong = attn(q, k, v).squeeze(0)

    dist_weak = (out_weak - v0).abs().sum()
    dist_strong = (out_strong - v0).abs().sum()
    assert dist_strong < dist_weak


def test_holographic_bool_mask_equals_truncated_keys():
    attn = HolographicAttention(temperature=2.0, attn_dropout=0.0)
    q = torch.randn(1, 2, 4, 4, dtype=torch.cfloat)
    k = torch.randn(1, 2, 5, 4, dtype=torch.cfloat)
    v = torch.randn(1, 2, 5, 4, dtype=torch.cfloat)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[:, 4:] = True
    out = attn(q, k, v, attn_mask=mask)
    expected = attn(q, k[..., :4, :], v[..., :4, :])
    torch.testing.assert_close(out, expected)


def test_holographic_float_mask_equals_bool_mask():
    attn = HolographicAttention(temperature=2.0, attn_dropout=0.0)
    q = torch.randn(1, 2, 4, 4, dtype=torch.cfloat)
    bool_mask = torch.zeros(4, 4, dtype=torch.bool)
    bool_mask[:, -1] = True
    float_mask = torch.zeros(4, 4)
    float_mask[:, -1] = float("-inf")
    torch.testing.assert_close(
        attn(q, q, q, attn_mask=bool_mask), attn(q, q, q, attn_mask=float_mask)
    )


def test_holographic_masked_coefficient_exactly_zero():
    """A masked weight of exactly 0 zeroes the complex coefficient too."""
    attn = HolographicAttention(temperature=2.0, attn_dropout=0.0)
    q = torch.randn(1, 1, 3, 4, dtype=torch.cfloat)
    k = torch.randn(1, 1, 3, 4, dtype=torch.cfloat)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[:, 0] = True
    # Make the masked value huge: any leak would visibly perturb the output.
    v = torch.randn(1, 1, 3, 4, dtype=torch.cfloat)
    v_huge = v.clone()
    v_huge[..., 0, :] = 1e30 + 1e30j
    torch.testing.assert_close(
        attn(q, k, v_huge, attn_mask=mask), attn(q, k, v, attn_mask=mask)
    )
