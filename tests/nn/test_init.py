"""Tests for complextorch.nn.init complex initializers."""

from __future__ import annotations

import math

import pytest
import torch

from complextorch.nn import init

# ---------- _get_fans + _check_complex via public functions ----------


def test_kaiming_rejects_real_tensor():
    with pytest.raises(TypeError, match="expects a complex tensor"):
        init.kaiming_normal_(torch.zeros(4, 4))


def test_kaiming_normal_fan_in():
    w = torch.empty(8, 16, dtype=torch.cfloat)
    init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")
    # Expected per-part std: sqrt(2) / sqrt(2 * 16) = 0.25
    assert w.is_complex()
    assert abs(w.real.std().item() - 0.25) < 0.1


def test_kaiming_normal_fan_out():
    w = torch.empty(8, 16, dtype=torch.cfloat)
    init.kaiming_normal_(w, mode="fan_out", nonlinearity="linear")
    assert torch.isfinite(w).all()


def test_kaiming_normal_3d_tensor_fan_calc():
    """tensor.dim() > 2 branch in _get_fans."""
    w = torch.empty(8, 4, 3, dtype=torch.cfloat)
    init.kaiming_normal_(w)
    assert w.shape == (8, 4, 3)


def test_kaiming_normal_1d_tensor_fan_calc():
    """tensor.dim() < 2 branch in _get_fans."""
    w = torch.empty(10, dtype=torch.cfloat)
    init.kaiming_normal_(w)
    assert w.shape == (10,)


def test_kaiming_uniform():
    w = torch.empty(8, 16, dtype=torch.cfloat)
    init.kaiming_uniform_(w)
    assert torch.isfinite(w).all()


def test_kaiming_invalid_nonlinearity_raises():
    w = torch.empty(4, 4, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="Unsupported nonlinearity"):
        init.kaiming_normal_(w, nonlinearity="bogus")


@pytest.mark.parametrize(
    "nl",
    [
        "linear",
        "tanh",
        "relu",
        "leaky_relu",
        "selu",
        "sigmoid",
        "conv1d",
        "conv2d",
        "conv3d",
    ],
)
def test_gain_branches_via_kaiming(nl):
    w = torch.empty(4, 4, dtype=torch.cfloat)
    init.kaiming_normal_(w, nonlinearity=nl)
    assert torch.isfinite(w).all()


# ---------- Xavier ----------


def test_xavier_normal():
    w = torch.empty(8, 16, dtype=torch.cfloat)
    init.xavier_normal_(w, gain=1.0)
    expected_std = 1.0 / math.sqrt(8 + 16)
    assert abs(w.real.std().item() - expected_std) < 0.1


def test_xavier_uniform():
    w = torch.empty(8, 16, dtype=torch.cfloat)
    init.xavier_uniform_(w)
    assert torch.isfinite(w).all()


# ---------- Trabelsi standard (polar Rayleigh) ----------


@pytest.mark.parametrize("kind", ["glorot", "xavier", "he", "kaiming"])
def test_trabelsi_standard_kinds(kind):
    w = torch.empty(16, 32, dtype=torch.cfloat)
    init.trabelsi_standard_(w, kind=kind)
    assert torch.isfinite(w).all()
    # Phases should span (-pi, pi) -> at least non-trivial spread
    assert w.angle().std().item() > 0.5


def test_trabelsi_standard_invalid_kind():
    w = torch.empty(4, 4, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="Unknown kind"):
        init.trabelsi_standard_(w, kind="bogus")


# ---------- Trabelsi independent (semi-unitary) ----------


@pytest.mark.parametrize("kind", ["glorot", "he"])
def test_trabelsi_independent_kinds(kind):
    w = torch.empty(8, 16, dtype=torch.cfloat)
    init.trabelsi_independent_(w, kind=kind)
    assert torch.isfinite(w).all()
    # Approximately semi-unitary scaled by `scale`: w @ w.conj().T = scale^2 * I
    prod = w @ w.conj().T
    diag = torch.diag(prod).real
    off = prod - torch.diag(torch.diag(prod))
    # diagonal entries should be ~equal
    assert (diag.std() / diag.mean()).item() < 0.1
    # off-diagonal should be small relative to diagonal
    assert off.abs().mean().item() < 0.1 * diag.mean().item()


@pytest.mark.parametrize("shape", [(64, 32), (32, 64)])
@pytest.mark.parametrize("kind", ["glorot", "he"])
def test_trabelsi_independent_variance_target(kind, shape):
    """Mean |w|^2 lands exactly on the paper's complex-Glorot/He target."""
    w = torch.empty(*shape, dtype=torch.cfloat)
    init.trabelsi_independent_(w, kind=kind)
    fan_in, fan_out = shape[1], shape[0]
    target = 2.0 / (fan_in + fan_out) if kind == "glorot" else 2.0 / fan_in
    assert abs(w.abs().pow(2).mean().item() - target) < 1e-3 * target


def test_trabelsi_independent_conv_shape_variance_target():
    """Conv-shaped tensor: fan_in = in * prod(kernel) = flat cols."""
    w = torch.empty(8, 4, 3, dtype=torch.cfloat)
    init.trabelsi_independent_(w)
    fan_in, fan_out = 4 * 3, 8 * 3
    target = 2.0 / (fan_in + fan_out)
    assert abs(w.abs().pow(2).mean().item() - target) < 1e-3 * target
    # The flattened (out, in*k) matrix stays exactly semi-unitary.
    flat = w.reshape(8, -1)
    prod = flat @ flat.conj().T
    scale_sq = prod.diagonal().real.mean()
    torch.testing.assert_close(
        prod, scale_sq * torch.eye(8, dtype=prod.dtype), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("shape", [(32, 8), (8, 32)])
def test_trabelsi_independent_exact_semi_unitarity(shape):
    """W^H W = s^2 I (tall) or W W^H = s^2 I (wide), exactly."""
    w = torch.empty(*shape, dtype=torch.cfloat)
    init.trabelsi_independent_(w)
    rows, cols = shape
    prod = w.conj().T @ w if rows >= cols else w @ w.conj().T
    scale_sq = prod.diagonal().real.mean()
    torch.testing.assert_close(
        prod,
        scale_sq * torch.eye(min(rows, cols), dtype=prod.dtype),
        atol=1e-5,
        rtol=0,
    )


def test_trabelsi_independent_square_is_unitary():
    """Square glorot: scale is exactly 1, so W is exactly unitary."""
    w = torch.empty(16, 16, dtype=torch.cfloat)
    init.trabelsi_independent_(w)
    torch.testing.assert_close(
        w.conj().T @ w, torch.eye(16, dtype=w.dtype), atol=1e-5, rtol=0
    )


def test_trabelsi_independent_invalid_kind():
    w = torch.empty(4, 4, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="Unknown kind"):
        init.trabelsi_independent_(w, kind="weird")


def test_trabelsi_independent_requires_2d():
    w = torch.empty(10, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="at least 2 dims"):
        init.trabelsi_independent_(w)
