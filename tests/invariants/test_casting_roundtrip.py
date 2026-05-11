"""Property tests: casting modules are inverses of each other."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from complextorch.nn.modules.casting import (
    ComplexToConcatenated,
    ComplexToInterleaved,
    ConcatenatedToComplex,
    InterleavedToComplex,
    RealToComplex,
)


@given(d=st.integers(1, 8))
@settings(max_examples=5, deadline=None)
def test_interleaved_roundtrip(d):
    z = torch.randn(3, d, dtype=torch.cfloat)
    back = InterleavedToComplex()(ComplexToInterleaved()(z))
    torch.testing.assert_close(back, z)


@given(d=st.integers(1, 8))
@settings(max_examples=5, deadline=None)
def test_concatenated_roundtrip(d):
    z = torch.randn(3, d, dtype=torch.cfloat)
    back = ConcatenatedToComplex()(ComplexToConcatenated()(z))
    torch.testing.assert_close(back, z)


@given(d=st.integers(1, 8))
@settings(max_examples=5, deadline=None)
def test_real_to_complex_zeros_imag(d):
    x = torch.randn(3, d)
    z = RealToComplex()(x)
    torch.testing.assert_close(z.real, x)
    torch.testing.assert_close(z.imag, torch.zeros_like(x))
