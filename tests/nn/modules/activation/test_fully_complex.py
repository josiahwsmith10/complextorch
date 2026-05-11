"""Tests for fully-complex activation functions."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.activation.fully_complex import (
    CVCardiod,
    CVSigLog,
    CVSigmoid,
    Mod,
    zReLU,
)


@pytest.fixture
def z():
    return torch.randn(8, dtype=torch.cfloat)


def test_cvsigmoid_forward(z):
    out = CVSigmoid()(z)
    assert out.shape == z.shape
    assert out.is_complex()


def test_zrelu_zeros_outside_first_quadrant():
    # In Q1 (angle in [0, pi/2]) -> passes; elsewhere -> 0
    z = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=torch.cfloat)
    out = zReLU()(z)
    assert out[0].abs().item() > 0  # Q1 -> passed
    assert out[1].abs().item() == 0  # Q2 -> zero
    assert out[2].abs().item() == 0  # Q3 -> zero
    assert out[3].abs().item() == 0  # Q4 -> zero


def test_cvcardiod_real_positive_axis_unchanged():
    z = torch.tensor([1.0 + 0j, 2.0 + 0j])
    out = CVCardiod()(z)
    # angle=0 -> 0.5*(1+cos(0))=1 -> output == z
    torch.testing.assert_close(out, z)


def test_cvcardiod_zeros_negative_real():
    z = torch.tensor([-1.0 + 0j, -3.0 + 0j])
    out = CVCardiod()(z)
    # angle=pi -> 0.5*(1+cos(pi))=0 -> output == 0
    torch.testing.assert_close(out, torch.zeros_like(z))


def test_cvsiglog_default_params(z):
    act = CVSigLog()
    out = act(z)
    assert out.shape == z.shape
    expected = z / (1.0 + z.abs() / 1.0)
    torch.testing.assert_close(out, expected)


def test_cvsiglog_custom_c_r(z):
    act = CVSigLog(c=2.0, r=3.0)
    out = act(z)
    expected = z / (2.0 + z.abs() / 3.0)
    torch.testing.assert_close(out, expected)


def test_mod_returns_magnitude(z):
    out = Mod()(z)
    torch.testing.assert_close(out, z.abs())
    assert not out.is_complex()
