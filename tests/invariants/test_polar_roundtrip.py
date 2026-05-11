"""Property tests: torch.polar(|z|, ∠z) ≡ z (away from |z|=0)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, strategies as st


@pytest.mark.parametrize("shape", [(4,), (2, 3), (1, 4, 5)])
@given(seed=st.integers(0, 10_000))
@settings(max_examples=5, deadline=None)
def test_polar_roundtrip(shape, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(*shape, generator=g, dtype=torch.cfloat) + 0.5
    back = torch.polar(z.abs(), z.angle())
    torch.testing.assert_close(back, z, atol=1e-5, rtol=1e-5)
