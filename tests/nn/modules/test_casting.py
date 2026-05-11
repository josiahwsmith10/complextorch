"""Tests for the layout-casting modules."""

from __future__ import annotations

import pytest
import torch

from complextorch.nn.modules.casting import (
    ComplexToConcatenated,
    ComplexToInterleaved,
    ConcatenatedToComplex,
    InterleavedToComplex,
    RealToComplex,
)


def test_interleaved_roundtrip():
    to_complex = InterleavedToComplex()
    to_real = ComplexToInterleaved()
    x = torch.randn(2, 6)  # last dim 6 = 2 * 3
    z = to_complex(x)
    assert z.shape == (2, 3)
    assert z.is_complex()
    x_back = to_real(z)
    torch.testing.assert_close(x_back, x)


def test_interleaved_to_complex_odd_dim_raises():
    with pytest.raises(ValueError, match="even"):
        InterleavedToComplex()(torch.randn(2, 5))


def test_complex_to_interleaved_real_input_raises():
    with pytest.raises(TypeError, match="expects a complex input"):
        ComplexToInterleaved()(torch.randn(2, 4))


def test_concatenated_roundtrip():
    to_complex = ConcatenatedToComplex()
    to_real = ComplexToConcatenated()
    z = torch.randn(2, 3, dtype=torch.cfloat)
    x = to_real(z)
    assert x.shape == (2, 6)
    z_back = to_complex(x)
    torch.testing.assert_close(z_back, z)


def test_concatenated_to_complex_odd_raises():
    with pytest.raises(ValueError, match="even"):
        ConcatenatedToComplex()(torch.randn(2, 5))


def test_complex_to_concatenated_real_input_raises():
    with pytest.raises(TypeError, match="expects a complex input"):
        ComplexToConcatenated()(torch.randn(2, 4))


def test_real_to_complex_from_real():
    lift = RealToComplex()
    x = torch.randn(2, 4)
    z = lift(x)
    assert z.is_complex()
    torch.testing.assert_close(z.real, x)
    torch.testing.assert_close(z.imag, torch.zeros_like(x))


def test_real_to_complex_complex_input_just_casts():
    lift = RealToComplex(dtype=torch.cdouble)
    x = torch.randn(2, 4, dtype=torch.cfloat)
    z = lift(x)
    assert z.dtype == torch.cdouble


def test_real_to_complex_extra_repr():
    s = RealToComplex().extra_repr()
    assert "dtype" in s
