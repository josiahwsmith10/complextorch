"""Tests for complex RMSNorm."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.rmsnorm import RMSNorm


def test_rmsnorm_int_shape():
    rms = RMSNorm(8)
    x = torch.randn(4, 8, dtype=torch.cfloat)
    out = rms(x)
    assert out.shape == x.shape


def test_rmsnorm_tuple_shape():
    rms = RMSNorm((4, 8))
    x = torch.randn(2, 4, 8, dtype=torch.cfloat)
    out = rms(x)
    assert out.shape == x.shape


def test_rmsnorm_list_shape():
    rms = RMSNorm([4, 8])
    x = torch.randn(2, 4, 8, dtype=torch.cfloat)
    out = rms(x)
    assert out.shape == x.shape


def test_rmsnorm_no_affine():
    rms = RMSNorm(8, elementwise_affine=False)
    x = torch.randn(4, 8, dtype=torch.cfloat)
    out = rms(x)
    assert rms.weight is None
    assert out.shape == x.shape


def test_rmsnorm_real_input_raises():
    rms = RMSNorm(8)
    with pytest.raises(TypeError, match="expects a complex input"):
        rms(torch.randn(4, 8))


def test_rmsnorm_extra_repr():
    s = RMSNorm(8).extra_repr()
    assert "normalized_shape=(8,)" in s


def test_rmsnorm_reset_parameters_no_op_no_affine():
    rms = RMSNorm(8, elementwise_affine=False)
    rms.reset_parameters()


def test_rmsnorm_unit_rms_when_no_affine():
    """Without affine, the RMS of |x|^2 over normalized dims should be ~1."""
    rms = RMSNorm(8, elementwise_affine=False)
    x = torch.randn(16, 8, dtype=torch.cfloat) * 5
    out = rms(x)
    rms_val = out.abs().pow(2).mean(dim=-1)
    torch.testing.assert_close(rms_val, torch.ones_like(rms_val), atol=0.01, rtol=0.01)
