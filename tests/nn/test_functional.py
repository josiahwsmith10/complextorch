"""Tests for complextorch.nn.functional primitives and norm helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from complextorch.nn import functional as F


# ---------- apply_complex (Gauss-trick lift) ----------


def test_apply_complex_matches_manual_real_imag():
    real_m = nn.Linear(4, 3, bias=False)
    imag_m = nn.Linear(4, 3, bias=False)
    x = torch.randn(2, 4, dtype=torch.cfloat)
    out = F.apply_complex(real_m, imag_m, x)
    expected = torch.complex(
        real_m(x.real) - imag_m(x.imag),
        real_m(x.imag) + imag_m(x.real),
    )
    torch.testing.assert_close(out, expected)
    assert out.is_complex()


# ---------- apply_complex_split ----------


def test_apply_complex_split_with_identity_returns_input():
    x = torch.randn(2, 5, dtype=torch.cfloat)
    out = F.apply_complex_split(lambda t: t, lambda t: t, x)
    torch.testing.assert_close(out, x)


def test_apply_complex_split_independent_functions():
    x = torch.randn(3, dtype=torch.cfloat)
    out = F.apply_complex_split(torch.relu, torch.tanh, x)
    expected = torch.complex(torch.relu(x.real), torch.tanh(x.imag))
    torch.testing.assert_close(out, expected)


# ---------- apply_complex_polar ----------


def test_apply_complex_polar_phase_none_preserves_phase():
    x = torch.randn(4, dtype=torch.cfloat)
    out = F.apply_complex_polar(torch.abs, None, x)
    torch.testing.assert_close(out.angle(), x.angle(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.abs(), x.abs().abs())  # mag_fun=abs is idempotent


def test_apply_complex_polar_phase_none_zero_magnitude_safe():
    """The clamp(min=1e-12) prevents division by zero at z=0."""
    x = torch.zeros(3, dtype=torch.cfloat)
    out = F.apply_complex_polar(torch.abs, None, x)
    assert torch.isfinite(out.real).all()
    assert torch.isfinite(out.imag).all()


def test_apply_complex_polar_with_phase_fun():
    x = torch.randn(4, dtype=torch.cfloat)
    out = F.apply_complex_polar(torch.abs, lambda p: p * 0, x)
    torch.testing.assert_close(
        out.imag, torch.zeros_like(out.imag), atol=1e-6, rtol=1e-6
    )


# ---------- inv_sqrtm2x2 ----------


def test_inv_sqrtm2x2_symmetric_gives_inverse_squareroot():
    a = torch.tensor([2.0])
    d = torch.tensor([3.0])
    b = torch.tensor([0.5])
    w, x, y, z = F.inv_sqrtm2x2(a, b, None, d, symmetric=True)
    assert y is None
    A = torch.tensor([[2.0, 0.5], [0.5, 3.0]])
    B = torch.tensor([[w.item(), x.item()], [x.item(), z.item()]])
    recovered = B @ B @ A
    torch.testing.assert_close(recovered, torch.eye(2), atol=1e-5, rtol=1e-5)


def test_inv_sqrtm2x2_non_symmetric_branch():
    a, b, c, d = (torch.tensor([v]) for v in (2.5, 0.1, 0.2, 3.0))
    w, x, y, z = F.inv_sqrtm2x2(a, b, c, d, symmetric=False)
    A = torch.tensor([[2.5, 0.1], [0.2, 3.0]])
    B = torch.tensor([[w.item(), x.item()], [y.item(), z.item()]])
    recovered = B @ B @ A
    torch.testing.assert_close(recovered, torch.eye(2), atol=1e-5, rtol=1e-5)


# ---------- whiten2x2_batch_norm ----------


def _stack_re_im(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=0)


def test_whiten2x2_batch_norm_training_no_running_stats():
    # Need many samples for the whitening to converge on identity covariance.
    z = torch.randn(256, 4, 16, dtype=torch.cfloat) * 3.0 + 0.5
    x = _stack_re_im(z)
    out = F.whiten2x2_batch_norm(x, training=True)
    assert out.shape == x.shape
    var = out.var(dim=(1, 2), unbiased=False)
    torch.testing.assert_close(var, torch.ones_like(var), atol=0.05, rtol=0.05)
    # Cross-covariance between real/imag should be near zero per feature.
    cov_ri = (out[0] * out[1]).mean(dim=(0, 2))
    assert cov_ri.abs().max().item() < 0.1


def test_whiten2x2_batch_norm_updates_running_stats():
    z = torch.randn(8, 4, 6, dtype=torch.cfloat)
    x = _stack_re_im(z)
    running_mean = torch.zeros(2, 4)
    running_cov = torch.eye(2).unsqueeze(-1).repeat(1, 1, 4) * 0.5
    rm_before = running_mean.clone()
    rc_before = running_cov.clone()
    F.whiten2x2_batch_norm(
        x, training=True, running_mean=running_mean, running_cov=running_cov
    )
    assert not torch.allclose(running_mean, rm_before)
    assert not torch.allclose(running_cov, rc_before)


def test_whiten2x2_batch_norm_eval_uses_running_stats():
    z_train = torch.randn(8, 4, 6, dtype=torch.cfloat)
    x_train = _stack_re_im(z_train)
    running_mean = torch.zeros(2, 4)
    running_cov = torch.eye(2).unsqueeze(-1).repeat(1, 1, 4).clone()
    F.whiten2x2_batch_norm(
        x_train, training=True, running_mean=running_mean, running_cov=running_cov
    )
    rm_after_train = running_mean.clone()
    rc_after_train = running_cov.clone()
    # Eval pass should NOT update the running stats
    z_eval = torch.randn(8, 4, 6, dtype=torch.cfloat)
    x_eval = _stack_re_im(z_eval)
    F.whiten2x2_batch_norm(
        x_eval, training=False, running_mean=running_mean, running_cov=running_cov
    )
    torch.testing.assert_close(running_mean, rm_after_train)
    torch.testing.assert_close(running_cov, rc_after_train)


# ---------- batch_norm wrapper ----------


def test_batch_norm_without_affine():
    z = torch.randn(8, 4, 10, dtype=torch.cfloat)
    out = F.batch_norm(z, training=True)
    assert out.is_complex()
    assert out.shape == z.shape


def test_batch_norm_with_affine():
    z = torch.randn(8, 4, 10, dtype=torch.cfloat)
    weight = torch.eye(2).unsqueeze(-1).repeat(1, 1, 4)
    bias = torch.zeros(2, 4)
    out = F.batch_norm(z, weight=weight, bias=bias, training=True)
    assert out.shape == z.shape


def test_batch_norm_with_running_stats_and_affine():
    z = torch.randn(8, 4, 10, dtype=torch.cfloat)
    running_mean = torch.zeros(2, 4)
    running_var = torch.eye(2).unsqueeze(-1).repeat(1, 1, 4)
    weight = torch.eye(2).unsqueeze(-1).repeat(1, 1, 4)
    bias = torch.zeros(2, 4)
    out = F.batch_norm(
        z,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=False,
    )
    assert out.shape == z.shape


# ---------- whiten2x2_layer_norm + layer_norm ----------


def test_layer_norm_without_affine():
    z = torch.randn(8, 16, dtype=torch.cfloat)
    out = F.layer_norm(z, normalized_shape=[16])
    assert out.shape == z.shape


def test_layer_norm_with_affine():
    z = torch.randn(8, 16, dtype=torch.cfloat)
    weight = torch.eye(2).unsqueeze(-1).repeat(1, 1, 16)
    bias = torch.zeros(2, 16)
    out = F.layer_norm(z, normalized_shape=[16], weight=weight, bias=bias)
    assert out.shape == z.shape


def test_layer_norm_multi_dim_normalized_shape():
    z = torch.randn(4, 8, 8, dtype=torch.cfloat)
    out = F.layer_norm(z, normalized_shape=[8, 8])
    assert out.shape == z.shape
