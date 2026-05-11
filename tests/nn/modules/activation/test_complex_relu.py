"""Tests for complex ReLU variants."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.activation.complex_relu import (
    CPReLU,
    CReLU,
    CVSplitReLU,
    zAbsReLU,
    zLeakyReLU,
)


@pytest.fixture
def z():
    return torch.randn(4, 8, dtype=torch.cfloat)


@pytest.mark.parametrize("cls", [CVSplitReLU, CReLU])
def test_split_relu_zeros_negatives(cls, z):
    out = cls(inplace=False)(z.clone())
    assert (out.real >= 0).all()
    assert (out.imag >= 0).all()


def test_cprelu_learnable(z):
    act = CPReLU()
    out = act(z)
    assert out.shape == z.shape
    # Two PReLU modules, each with a learnable weight
    params = list(act.parameters())
    assert len(params) == 2


def test_zabsrelu_threshold_zero_passes_all():
    z = torch.randn(8, dtype=torch.cfloat)
    out = zAbsReLU(a_init=0.0)(z)
    torch.testing.assert_close(out, z)


def test_zabsrelu_high_threshold_zeros_small():
    z = torch.tensor([0.1 + 0j, 5.0 + 0j], dtype=torch.cfloat)
    out = zAbsReLU(a_init=1.0)(z)
    assert out[0].abs().item() == 0.0
    assert out[1].abs().item() == 5.0


def test_zabsrelu_real_input_path():
    """Test the non-complex branch in forward."""
    x = torch.tensor([0.1, 5.0])  # real, not complex
    out = zAbsReLU(a_init=1.0)(x)
    assert out[0].abs().item() == 0.0


def test_zleakyrelu_first_quadrant_passes():
    z = torch.tensor([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=torch.cfloat)
    out = zLeakyReLU(negative_slope=0.5)(z)
    torch.testing.assert_close(out[0], z[0])
    torch.testing.assert_close(out[1], 0.5 * z[1])
    torch.testing.assert_close(out[2], 0.5 * z[2])
    torch.testing.assert_close(out[3], 0.5 * z[3])


def test_zleakyrelu_extra_repr():
    s = zLeakyReLU(negative_slope=0.2).extra_repr()
    assert "0.2" in s
