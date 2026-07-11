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


def test_rmsnorm_affine_init_is_identity():
    """Regression: the affine used to init at 0.7071*I, halving output power."""
    rms = RMSNorm(8)
    eye = torch.eye(2).view(2, 2, 1).expand(2, 2, 8)
    torch.testing.assert_close(rms.weight.data, eye.contiguous())
    # Unit power at init, exactly like the affine-free path.
    x = torch.randn(64, 8, dtype=torch.cfloat) * 3
    power = rms(x).abs().pow(2).mean(dim=-1)
    torch.testing.assert_close(power, torch.ones_like(power), atol=0.01, rtol=0.01)


def test_rmsnorm_matches_torch_rmsnorm_at_init_on_real_input():
    """At init, agrees with torch.nn.RMSNorm on a real-valued cfloat input."""
    rms = RMSNorm(8, eps=1e-6)
    ref = torch.nn.RMSNorm(8, eps=1e-6)
    x = torch.randn(4, 8)
    out = rms(x.to(torch.cfloat))
    # Real input has |z|^2 = x^2, so both compute x / sqrt(mean(x^2) + eps).
    torch.testing.assert_close(out.real, ref(x))
    torch.testing.assert_close(out.imag, torch.zeros_like(x))
