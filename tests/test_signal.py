"""Tests for complextorch.signal.pwelch."""

from __future__ import annotations

import pytest
import torch

from complextorch.signal import analytic_signal, hilbert, pwelch


def test_pwelch_real_default_onesided():
    x = torch.randn(2048)
    freqs, psd = pwelch(x, window=256, fs=1.0)
    assert freqs.shape == (129,)
    assert psd.shape == (129,)
    assert torch.all(psd >= 0)


def test_pwelch_complex_two_sided():
    x = torch.randn(2048, dtype=torch.cfloat)
    freqs, psd = pwelch(x, window=256, fs=2.0)
    assert freqs.shape == (256,)
    assert psd.shape == (256,)


def test_pwelch_user_window_tensor():
    x = torch.randn(512)
    win = torch.hann_window(64)
    freqs, psd = pwelch(x, window=win, fs=1.0)
    assert freqs.shape[0] == 33
    assert psd.shape[0] == 33


def test_pwelch_scaling_spectrum():
    x = torch.randn(1024)
    _, psd_density = pwelch(x, window=128, scaling="density")
    _, psd_spectrum = pwelch(x, window=128, scaling="spectrum")
    assert psd_density.shape == psd_spectrum.shape
    assert not torch.allclose(psd_density, psd_spectrum)


def test_pwelch_detrend_none():
    x = torch.randn(512) + 10.0  # large DC offset
    _, psd_none = pwelch(x, window=64, detrend="none")
    _, psd_const = pwelch(x, window=64, detrend="constant")
    assert psd_none[0] > psd_const[0]  # detrending removes DC spike


def test_pwelch_overlap_eq_window_raises():
    with pytest.raises(ValueError, match="must be smaller than window length"):
        pwelch(torch.randn(64), window=32, n_overlap=32)


def test_pwelch_invalid_detrend_raises():
    with pytest.raises(ValueError, match="detrend must be"):
        pwelch(torch.randn(64), window=16, detrend="linear")


def test_pwelch_invalid_scaling_raises():
    with pytest.raises(ValueError, match="scaling must be"):
        pwelch(torch.randn(64), window=16, scaling="bogus")


def test_pwelch_onesided_with_complex_raises():
    x = torch.randn(128, dtype=torch.cfloat)
    with pytest.raises(ValueError, match="invalid for complex signals"):
        pwelch(x, window=16, return_onesided=True)


def test_pwelch_window_longer_than_signal():
    """Window length is clamped to signal length."""
    x = torch.randn(8)
    freqs, _psd = pwelch(x, window=64)
    assert freqs.shape[0] == 5  # rfft(8) → 5


def test_pwelch_batched_input():
    x = torch.randn(4, 3, 512)
    freqs, psd = pwelch(x, window=64, fs=1.0)
    assert psd.shape == (4, 3, 33)
    assert freqs.shape == (33,)


def test_pwelch_two_freq_bins_path():
    """Hit the elif psd.shape[-1] == 2 branch (win_len=2 → rfft returns 2 bins)."""
    x = torch.randn(16)
    freqs, psd = pwelch(x, window=2, n_overlap=0)
    assert freqs.shape[0] == 2
    assert psd.shape[0] == 2


def test_pwelch_differentiable():
    x = torch.randn(256, requires_grad=True)
    _, psd = pwelch(x, window=64)
    psd.sum().backward()
    assert x.grad is not None


# -------- analytic signal / Hilbert transform --------


@pytest.mark.parametrize("n", [64, 65])  # even and odd length branches
def test_analytic_signal_of_integer_cycle_cosine(n):
    # cos(2*pi*k*t/N) with integer k -> analytic signal is exactly exp(j*2*pi*k*t/N).
    k = 5
    t = torch.arange(n, dtype=torch.float64)
    phase = 2 * torch.pi * k * t / n
    za = analytic_signal(torch.cos(phase))
    expected = torch.complex(torch.cos(phase), torch.sin(phase))
    torch.testing.assert_close(za, expected, atol=1e-9, rtol=1e-9)


def test_hilbert_of_cosine_is_sine():
    n, k = 128, 7
    t = torch.arange(n, dtype=torch.float64)
    phase = 2 * torch.pi * k * t / n
    torch.testing.assert_close(
        hilbert(torch.cos(phase)), torch.sin(phase), atol=1e-9, rtol=1e-9
    )


def test_analytic_signal_real_part_preserved_and_dim():
    x = torch.randn(3, 32)
    za = analytic_signal(x, dim=-1)
    assert za.shape == x.shape
    assert za.is_complex()
    torch.testing.assert_close(za.real, x, atol=1e-5, rtol=1e-5)


def test_analytic_signal_differentiable():
    x = torch.randn(64, requires_grad=True)
    analytic_signal(x).imag.sum().backward()
    assert x.grad is not None
