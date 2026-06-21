"""Tests for HolographicAttention (interference-aware complex attention)."""

from __future__ import annotations

import torch

from complextorch.nn.modules.attention.holographic import HolographicAttention


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
