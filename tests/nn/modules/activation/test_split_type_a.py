"""Tests for split Type-A activation functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from complextorch.nn.modules.activation.split_type_A import (
    CCELU,
    CELU,
    CGELU,
    CSigmoid,
    CTanh,
    CVSplitAbs,
    CVSplitCELU,
    CVSplitELU,
    CVSplitGELU,
    CVSplitSigmoid,
    CVSplitTanh,
    GeneralizedSplitActivation,
)


@pytest.fixture
def z():
    return torch.randn(4, 8, dtype=torch.cfloat)


@pytest.mark.parametrize(
    "cls",
    [
        CVSplitTanh,
        CTanh,
        CVSplitSigmoid,
        CSigmoid,
        CVSplitAbs,
        CVSplitELU,
        CELU,
        CVSplitCELU,
        CCELU,
        CVSplitGELU,
        CGELU,
    ],
)
def test_split_type_a_forward_preserves_shape_and_complex_dtype(cls, z):
    act = cls()
    out = act(z)
    assert out.shape == z.shape
    assert out.is_complex()


def test_generalized_split_activation_user_provided_modules(z):
    act = GeneralizedSplitActivation(nn.ReLU(), nn.Tanh())
    out = act(z)
    expected = torch.complex(torch.relu(z.real), torch.tanh(z.imag))
    torch.testing.assert_close(out, expected)


def test_cvsplit_abs_outputs_absolute_values(z):
    out = CVSplitAbs()(z)
    torch.testing.assert_close(out.real, z.real.abs())
    torch.testing.assert_close(out.imag, z.imag.abs())


def test_cvsplit_elu_with_alpha_inplace(z):
    act = CVSplitELU(alpha=0.5, inplace=False)
    out = act(z)
    assert out.shape == z.shape


def test_cvsplit_celu_with_alpha(z):
    act = CVSplitCELU(alpha=2.0, inplace=False)
    out = act(z)
    assert out.shape == z.shape


def test_cvsplit_gelu_tanh_approximate(z):
    act = CVSplitGELU(approximate="tanh")
    out = act(z)
    assert out.shape == z.shape
