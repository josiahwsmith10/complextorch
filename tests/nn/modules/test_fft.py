"""Tests for FFTBlock / IFFTBlock."""

from __future__ import annotations

import torch

from complextorch.nn.modules.fft import FFTBlock, IFFTBlock


def test_fft_round_trip():
    fwd = FFTBlock(dim=-1, norm="ortho")
    inv = IFFTBlock(dim=-1, norm="ortho")
    x = torch.randn(2, 8, dtype=torch.cfloat)
    torch.testing.assert_close(inv(fwd(x)), x, atol=1e-5, rtol=1e-5)


def test_fft_with_n_param():
    fwd = FFTBlock(n=16, dim=-1)
    x = torch.randn(2, 8, dtype=torch.cfloat)
    out = fwd(x)
    assert out.shape == (2, 16)


def test_ifft_with_n_param():
    inv = IFFTBlock(n=8, dim=-1)
    x = torch.randn(2, 16, dtype=torch.cfloat)
    out = inv(x)
    assert out.shape == (2, 8)
