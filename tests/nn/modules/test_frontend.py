"""Tests for learnable complex time-frequency front-ends."""

from __future__ import annotations

import torch
import torch.nn as nn

from complextorch.nn.modules.frontend import (
    STFT,
    ComplexGaborConv1d,
    InverseSTFT,
    MorletConv1d,
)

# ---------- STFT / InverseSTFT ----------


def test_stft_shape_and_complex_output():
    stft = STFT(n_fft=16, hop_length=4)
    x = torch.randn(2, 64, dtype=torch.cfloat)
    spec = stft(x)
    assert spec.shape == (2, 16, 13)  # (n_fft, n_frames); (64-16)//4 + 1 = 13
    assert spec.is_complex()


def test_stft_real_input_casts_to_complex():
    spec = STFT(16)(torch.randn(2, 64))
    assert spec.is_complex()


def test_stft_istft_roundtrip_interior():
    stft = STFT(n_fft=16, hop_length=4)
    istft = InverseSTFT(n_fft=16, hop_length=4)  # windows init identically (Hann)
    x = torch.randn(2, 64, dtype=torch.cfloat)
    recon = istft(stft(x))
    assert recon.shape[-1] == 64
    torch.testing.assert_close(recon[..., 20:44], x[..., 20:44], atol=1e-4, rtol=1e-4)


def test_stft_istft_roundtrip_with_tied_trained_window():
    # Tying the synthesis window to the analysis window keeps the inverse exact
    # even after the (now shared) learnable window drifts away from Hann.
    stft = STFT(n_fft=16, hop_length=4)
    istft = InverseSTFT(n_fft=16, hop_length=4)
    istft.window = stft.window  # share the learnable window
    with torch.no_grad():
        stft.window.add_(0.2 * torch.randn_like(stft.window))
    x = torch.randn(2, 64, dtype=torch.cfloat)
    recon = istft(stft(x))
    torch.testing.assert_close(recon[..., 20:44], x[..., 20:44], atol=1e-4, rtol=1e-4)


def test_stft_grad_flows_into_window():
    stft = STFT(16, hop_length=4)
    x = torch.randn(2, 64, dtype=torch.cfloat)
    stft(x).abs().sum().backward()
    assert stft.window.grad is not None


def test_istft_grad_flows_into_window():
    istft = InverseSTFT(16, hop_length=4)
    spec = torch.randn(2, 16, 5, dtype=torch.cfloat)
    istft(spec).abs().sum().backward()
    assert istft.window.grad is not None


def test_stft_fixed_window_is_buffer():
    stft = STFT(16, learnable_window=False)
    istft = InverseSTFT(16, learnable_window=False)
    assert not isinstance(stft.window, nn.Parameter)
    assert not isinstance(istft.window, nn.Parameter)
    x = torch.randn(1, 64, dtype=torch.cfloat)
    assert istft(stft(x)).is_complex()


def test_stft_extra_repr():
    assert "n_fft=16" in STFT(16, hop_length=4).extra_repr()
    assert "n_fft=16" in InverseSTFT(16, hop_length=4).extra_repr()


# ---------- Gabor / Morlet filterbanks ----------


def test_gabor_forward_shape_and_complex():
    conv = ComplexGaborConv1d(2, 8, kernel_size=15, padding=7)
    x = torch.randn(3, 2, 64, dtype=torch.cfloat)
    y = conv(x)
    assert y.shape == (3, 8, 64)
    assert y.is_complex()


def test_gabor_real_input_casts():
    conv = ComplexGaborConv1d(1, 4, kernel_size=9, padding=4)
    y = conv(torch.randn(2, 1, 32))
    assert y.is_complex()


def test_gabor_grad_flows_into_params():
    conv = ComplexGaborConv1d(2, 8, kernel_size=15, padding=7)
    x = torch.randn(3, 2, 64, dtype=torch.cfloat)
    conv(x).abs().sum().backward()
    assert conv.freq.grad is not None
    assert conv.log_sigma.grad is not None


def test_gabor_extra_repr():
    assert "out_channels=8" in ComplexGaborConv1d(2, 8, 15).extra_repr()


def test_morlet_kernels_are_zero_mean():
    mor = MorletConv1d(1, 4, kernel_size=21)
    k = mor._kernels()
    torch.testing.assert_close(
        k.sum(-1), torch.zeros(4, dtype=torch.cfloat), atol=1e-5, rtol=0
    )


def test_morlet_forward_complex():
    y = MorletConv1d(1, 4, kernel_size=9, padding=4)(
        torch.randn(2, 1, 32, dtype=torch.cfloat)
    )
    assert y.is_complex()
