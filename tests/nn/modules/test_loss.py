"""Tests for complex loss functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from complextorch.nn.modules.loss import (
    SSIM,
    AnalyticSignalLoss,
    CVCauchyError,
    CVFourthPowError,
    CVLogCoshError,
    CVLogError,
    CVQuadError,
    GeneralizedPolarLoss,
    GeneralizedSplitLoss,
    HolographicReconstructionLoss,
    MSELoss,
    PerpLossSSIM,
    SplitL1,
    SplitMSE,
    SplitSSIM,
    phase_smoothness,
)
from complextorch.signal import analytic_signal

# -------- _reduce branches via parameterized losses --------

REDUCTION_LOSSES = [
    CVQuadError,
    CVFourthPowError,
    CVCauchyError,
    CVLogCoshError,
    MSELoss,
    HolographicReconstructionLoss,
]


@pytest.mark.parametrize("cls", REDUCTION_LOSSES)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_reduction_branches(cls, reduction):
    x = torch.randn(4, 8, dtype=torch.cfloat)
    y = torch.randn(4, 8, dtype=torch.cfloat)
    loss = cls(reduction=reduction)
    out = loss(x, y)
    if reduction == "none":
        assert out.shape == x.shape
    else:
        assert out.dim() == 0


@pytest.mark.parametrize("cls", REDUCTION_LOSSES)
def test_invalid_reduction(cls):
    x = torch.randn(2, 4, dtype=torch.cfloat)
    y = torch.randn(2, 4, dtype=torch.cfloat)
    loss = cls(reduction="bogus")
    with pytest.raises(ValueError, match="reduction must be"):
        loss(x, y)


# -------- CVLogError (separate because log() requires non-zero input) --------


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cvlogerror_reduction(reduction):
    x = torch.randn(4, dtype=torch.cfloat) + 1.0
    y = torch.randn(4, dtype=torch.cfloat) + 1.0
    out = CVLogError(reduction=reduction)(x, y)
    assert torch.isfinite(out).all() or reduction == "none"


def test_cvlogerror_invalid_reduction():
    x = torch.randn(2, dtype=torch.cfloat) + 1.0
    y = torch.randn(2, dtype=torch.cfloat) + 1.0
    with pytest.raises(ValueError, match="reduction must be"):
        CVLogError(reduction="bogus")(x, y)


# -------- Split losses --------


def test_split_l1():
    x = torch.randn(2, 8, dtype=torch.cfloat)
    y = torch.randn(2, 8, dtype=torch.cfloat)
    out = SplitL1()(x, y)
    expected = nn.L1Loss()(x.real, y.real) + nn.L1Loss()(x.imag, y.imag)
    torch.testing.assert_close(out, expected)


def test_split_mse():
    x = torch.randn(2, 8, dtype=torch.cfloat)
    y = torch.randn(2, 8, dtype=torch.cfloat)
    out = SplitMSE()(x, y)
    expected = nn.MSELoss()(x.real, y.real) + nn.MSELoss()(x.imag, y.imag)
    torch.testing.assert_close(out, expected)


def test_generalized_split_loss_user_provided():
    loss = GeneralizedSplitLoss(nn.L1Loss(), nn.MSELoss())
    x = torch.randn(2, 8, dtype=torch.cfloat)
    y = torch.randn(2, 8, dtype=torch.cfloat)
    out = loss(x, y)
    expected = nn.L1Loss()(x.real, y.real) + nn.MSELoss()(x.imag, y.imag)
    torch.testing.assert_close(out, expected)


# -------- Polar loss --------


def test_generalized_polar_loss():
    loss = GeneralizedPolarLoss(
        nn.MSELoss(), nn.MSELoss(), weight_mag=2.0, weight_phase=0.5
    )
    x = torch.randn(2, 8, dtype=torch.cfloat)
    y = torch.randn(2, 8, dtype=torch.cfloat)
    out = loss(x, y)
    expected = 2.0 * nn.MSELoss()(x.abs(), y.abs()) + 0.5 * nn.MSELoss()(
        x.angle(), y.angle()
    )
    torch.testing.assert_close(out, expected)


# -------- SSIM family --------


def test_ssim_default_reduction():
    s = SSIM()
    x = torch.randn(1, 1, 32, 32)
    y = torch.randn(1, 1, 32, 32)
    out = s(x, y)
    assert out.dim() == 0


def test_ssim_full():
    s = SSIM()
    x = torch.randn(1, 1, 32, 32)
    y = torch.randn(1, 1, 32, 32)
    out = s(x, y, full=True)
    assert out.dim() == 4


def test_ssim_with_data_range():
    s = SSIM()
    x = torch.randn(1, 1, 32, 32)
    y = torch.randn(1, 1, 32, 32)
    data_range = torch.ones(1)
    out = s(x, y, data_range=data_range)
    assert out.dim() == 0


def test_split_ssim():
    s = SplitSSIM()
    x = torch.randn(1, 1, 32, 32, dtype=torch.cfloat)
    y = torch.randn(1, 1, 32, 32, dtype=torch.cfloat)
    out = s(x, y)
    assert out.dim() == 0


def test_perp_loss_ssim():
    loss = PerpLossSSIM()
    x = torch.randn(1, 1, 32, 32, dtype=torch.cfloat)
    y = torch.randn(1, 1, 32, 32, dtype=torch.cfloat)
    out = loss(x, y)
    assert torch.isfinite(out).all()


def test_cvcauchy_with_custom_c():
    loss = CVCauchyError(c=2.0, reduction="mean")
    x = torch.randn(4, dtype=torch.cfloat)
    y = torch.randn(4, dtype=torch.cfloat)
    out = loss(x, y)
    assert out.dim() == 0


# -------- Holographic reconstruction loss --------


def test_holographic_reconstruction_matches_abs_sq():
    x = torch.randn(3, 5, dtype=torch.cfloat)
    y = torch.randn(3, 5, dtype=torch.cfloat)
    out = HolographicReconstructionLoss(reduction="sum")(x, y)
    torch.testing.assert_close(out, ((x - y).abs() ** 2).sum())


def test_holographic_reconstruction_zero_on_identity():
    x = torch.randn(3, 5, dtype=torch.cfloat)
    out = HolographicReconstructionLoss()(x, x)
    torch.testing.assert_close(out, torch.zeros(()))


# -------- phase_smoothness regularizer --------


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_phase_smoothness_reduction(reduction):
    x = torch.randn(2, 6, dtype=torch.cfloat)
    out = phase_smoothness(x, dim=-1, reduction=reduction)
    if reduction == "none":
        assert out.shape == (2, 5)  # one fewer along the diff dim
    else:
        assert out.dim() == 0


def test_phase_smoothness_zero_on_constant_phase():
    # Constant phase along the sequence -> zero variation.
    mag = torch.rand(1, 8) + 0.1
    x = torch.polar(mag, torch.full((1, 8), 0.4))
    out = phase_smoothness(x, dim=-1)
    torch.testing.assert_close(out, torch.zeros(()), atol=1e-6, rtol=0)


def test_phase_smoothness_invalid_reduction():
    x = torch.randn(2, 6, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="reduction must be"):
        phase_smoothness(x, reduction="bogus")


# -------- AnalyticSignalLoss --------


def test_analytic_signal_loss_zero_on_true_analytic_signal():
    x = torch.randn(2, 64)
    z = analytic_signal(x)  # imag(z) == hilbert(real(z)) by construction
    out = AnalyticSignalLoss()(z)
    torch.testing.assert_close(out, torch.zeros(()), atol=1e-10, rtol=0)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_analytic_signal_loss_reduction(reduction):
    z = torch.randn(2, 16, dtype=torch.cfloat)
    out = AnalyticSignalLoss(reduction=reduction)(z)
    if reduction == "none":
        assert out.shape == z.shape
    else:
        assert out.dim() == 0


def test_analytic_signal_loss_invalid_reduction():
    z = torch.randn(2, 16, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="reduction must be"):
        AnalyticSignalLoss(reduction="bogus")(z)
