"""Tests for complex diagonal state-space models (S4D / DSS / Mamba)."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.ssm import DSS, S4D, MambaBlock, S4DBlock


@pytest.mark.parametrize("cls", [S4D, DSS])
def test_ssm_forward_shape_and_dtype(cls):
    m = cls(channels=4, state_size=8)
    u = torch.randn(2, 12, 4, dtype=torch.cfloat)
    y = m(u)
    assert y.shape == u.shape
    assert y.is_complex()


@pytest.mark.parametrize("cls", [S4D, DSS])
def test_ssm_grad_flows(cls):
    m = cls(channels=3, state_size=6)
    u = torch.randn(2, 10, 3, dtype=torch.cfloat)
    (m(u).abs() ** 2).sum().backward()
    for p in (m.log_dt, m.C, m.log_neg_A_real, m.D):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_ssm_extra_repr():
    s = S4D(channels=4, state_size=8).extra_repr()
    assert "channels=4" in s
    assert "state_size=8" in s


# ---------- S4DBlock ----------


@pytest.mark.parametrize("variant", ["s4d", "dss"])
def test_s4dblock_forward(variant):
    block = S4DBlock(channels=6, state_size=8, variant=variant)
    x = torch.randn(2, 9, 6, dtype=torch.cfloat)
    out = block(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_s4dblock_invalid_variant():
    with pytest.raises(ValueError, match="variant must be"):
        S4DBlock(channels=6, variant="bogus")


# ---------- MambaBlock ----------


def test_mamba_forward_shape():
    block = MambaBlock(channels=8, state_size=4)
    x = torch.randn(2, 7, 8, dtype=torch.cfloat)
    out = block(x)
    assert out.shape == x.shape
    assert out.is_complex()


def test_mamba_explicit_dt_rank_and_grad():
    block = MambaBlock(channels=8, state_size=4, dt_rank=3)
    assert block.dt_rank == 3
    x = torch.randn(2, 6, 8, dtype=torch.cfloat)
    (block(x).abs() ** 2).sum().backward()
    assert block.log_neg_A_real.grad is not None
    assert torch.isfinite(block.log_neg_A_real.grad).all()


def test_mamba_is_input_dependent():
    """A selective SSM's later output reacts to an earlier-timestep change."""
    block = MambaBlock(channels=8, state_size=4)
    x = torch.randn(1, 8, 8, dtype=torch.cfloat)
    x2 = x.clone()
    x2[:, 0] += 1.0  # perturb only the first timestep
    y = block(x)
    y2 = block(x2)
    assert not torch.allclose(y[:, -1], y2[:, -1])


def test_mamba_extra_repr():
    s = MambaBlock(channels=8, state_size=4).extra_repr()
    assert "d_inner=16" in s
    assert "state_size=4" in s
