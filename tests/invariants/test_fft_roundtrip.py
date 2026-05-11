"""Property tests: IFFT(FFT(x)) ≡ x."""

from __future__ import annotations

import torch
from hypothesis import given, settings, strategies as st

from complextorch.nn.modules.fft import FFTBlock, IFFTBlock


@given(n=st.integers(4, 32))
@settings(max_examples=5, deadline=None)
def test_fft_ifft_roundtrip(n):
    fwd = FFTBlock(dim=-1, norm="ortho")
    inv = IFFTBlock(dim=-1, norm="ortho")
    x = torch.randn(2, n, dtype=torch.cfloat)
    torch.testing.assert_close(inv(fwd(x)), x, atol=1e-4, rtol=1e-4)
