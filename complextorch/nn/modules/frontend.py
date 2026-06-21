r"""
Learnable Complex Time-Frequency Front-Ends
===========================================

Differentiable signal front-ends that turn a (real or complex) 1-D signal into
a native complex time-frequency representation, with learnable parameters so the
front-end can be trained end-to-end with the rest of the network.

- :class:`STFT` / :class:`InverseSTFT` -- short-time Fourier transform with a
  **learnable analysis/synthesis window**. The inverse uses a window-squared
  overlap-add normalisation, so ``InverseSTFT(STFT(x)) == x`` (on the covered,
  non-edge region) for any non-vanishing window **provided the synthesis window
  matches the analysis window** -- tie them with ``istft.window = stft.window``
  (the two modules default to identical Hann windows, but learnable windows
  diverge once trained unless tied).
- :class:`ComplexGaborConv1d` / :class:`MorletConv1d` -- learnable complex
  Gabor / Morlet filterbanks (a complex, wavelet-style analogue of SincNet):
  each filter is a windowed complex exponential with a learnable centre
  frequency and bandwidth, applied with a complex 1-D convolution.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["STFT", "ComplexGaborConv1d", "InverseSTFT", "MorletConv1d"]


class STFT(nn.Module):
    r"""
    Learnable Short-Time Fourier Transform
    --------------------------------------

    Frames the last dimension of ``input`` into overlapping windows, applies a
    (learnable) window, and FFTs each frame. Works on real or complex input and
    always returns a **complex** spectrogram of shape
    ``(..., n_fft, n_frames)`` (frequency x time, two-sided).

    Args:
        n_fft: frame length / number of frequency bins.
        hop_length: hop between frames (defaults to ``n_fft // 2``).
        learnable_window: if ``True`` the window is a learnable parameter
            (initialised to a Hann window); otherwise it is a fixed buffer.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int | None = None,
        learnable_window: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 2
        win = torch.hann_window(n_fft)
        if learnable_window:
            self.window = nn.Parameter(win)
        else:
            self.register_buffer("window", win)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Compute the complex STFT.

        Args:
            input (torch.Tensor): ``(..., T)`` real or complex signal.

        Returns:
            torch.Tensor: complex ``(..., n_fft, n_frames)`` spectrogram.
        """
        frames = input.unfold(-1, self.n_fft, self.hop_length)  # (..., F, n_fft)
        frames = frames * self.window
        spec = torch.fft.fft(frames, n=self.n_fft, dim=-1)  # (..., n_frames, n_fft)
        return spec.transpose(-2, -1)  # (..., n_fft, n_frames)

    def extra_repr(self) -> str:
        return f"n_fft={self.n_fft}, hop_length={self.hop_length}"


class InverseSTFT(nn.Module):
    r"""
    Inverse of :class:`STFT` (window-squared overlap-add)
    -----------------------------------------------------

    Inverts a complex spectrogram of shape ``(..., n_fft, n_frames)`` back to a
    time-domain signal via IFFT + windowed overlap-add, normalised by the
    overlap-added squared window. This reconstructs the original signal exactly
    on every output sample covered by at least one non-zero window tap **as long
    as the synthesis window equals the analysis window** used by the forward
    :class:`STFT`.

    .. note::

        :class:`STFT` and :class:`InverseSTFT` own **separate** windows. They
        default to identical Hann windows, so an untrained pair already inverts
        exactly; but two *learnable* windows diverge during training. Tie them
        to keep the reconstruction exact::

            stft = STFT(n_fft); istft = InverseSTFT(n_fft)
            istft.window = stft.window   # share the (learnable) window

    Args:
        n_fft: frame length (must match the forward :class:`STFT`).
        hop_length: hop between frames (defaults to ``n_fft // 2``).
        learnable_window: see :class:`STFT`.
        eps: stabiliser for the window-power normalisation.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int | None = None,
        learnable_window: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 2
        self.eps = eps
        win = torch.hann_window(n_fft)
        if learnable_window:
            self.window = nn.Parameter(win)
        else:
            self.register_buffer("window", win)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        r"""Reconstruct the time-domain signal.

        Args:
            spec (torch.Tensor): complex ``(..., n_fft, n_frames)`` spectrogram.

        Returns:
            torch.Tensor: complex ``(..., T)`` signal.
        """
        frames = spec.transpose(-2, -1)  # (..., n_frames, n_fft)
        frames = torch.fft.ifft(frames, n=self.n_fft, dim=-1)
        frames = frames * self.window  # synthesis window

        n_frames = frames.shape[-2]
        hop, n_fft = self.hop_length, self.n_fft
        out_len = (n_frames - 1) * hop + n_fft
        batch = frames.shape[:-2]

        signal = torch.zeros(*batch, out_len, dtype=frames.dtype, device=frames.device)
        wsq = torch.zeros(out_len, dtype=self.window.real.dtype, device=frames.device)
        win_sq = self.window.abs() ** 2
        for m in range(n_frames):
            start = m * hop
            pad_l = torch.zeros(*batch, start, dtype=frames.dtype, device=frames.device)
            pad_r = torch.zeros(
                *batch,
                out_len - start - n_fft,
                dtype=frames.dtype,
                device=frames.device,
            )
            signal = signal + torch.cat([pad_l, frames[..., m, :], pad_r], dim=-1)
            wsq = wsq + F.pad(win_sq, (start, out_len - start - n_fft))
        return signal / (wsq + self.eps)

    def extra_repr(self) -> str:
        return f"n_fft={self.n_fft}, hop_length={self.hop_length}"


class ComplexGaborConv1d(nn.Module):
    r"""
    Learnable Complex Gabor Filterbank (1-D)
    ----------------------------------------

    A 1-D convolution whose kernels are complex Gabor atoms

    .. math::

        g_o(t) = e^{-t^2 / (2\sigma_o^2)}\; e^{j 2\pi f_o t},

    with a learnable centre frequency :math:`f_o` and bandwidth :math:`\sigma_o`
    per output channel (shared across input channels, SincNet-style). Produces a
    complex output.

    Args:
        in_channels: number of input channels.
        out_channels: number of filters.
        kernel_size: filter length (samples).
        stride: convolution stride.
        padding: convolution padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.freq = nn.Parameter(torch.linspace(0.05, 0.45, out_channels))
        self.log_sigma = nn.Parameter(
            torch.full((out_channels,), math.log(kernel_size / 6.0))
        )
        t = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        self.register_buffer("t", t)

    def _env_carrier(self) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.t.unsqueeze(0)  # (1, K)
        sigma = torch.exp(self.log_sigma).unsqueeze(1)  # (O, 1)
        env = torch.exp(-0.5 * (t / sigma) ** 2)  # (O, K)
        phase = 2 * math.pi * self.freq.unsqueeze(1) * t  # (O, K)
        carrier = torch.polar(torch.ones_like(phase), phase)  # (O, K) complex
        return env, carrier

    def _kernels(self) -> torch.Tensor:
        env, carrier = self._env_carrier()
        return env * carrier  # (O, K) complex

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Apply the complex filterbank.

        Args:
            input (torch.Tensor): ``(B, in_channels, T)`` real or complex signal.

        Returns:
            torch.Tensor: complex ``(B, out_channels, T')`` feature map.
        """
        kern = self._kernels()  # (O, K)
        weight = (
            kern.unsqueeze(1)
            .expand(self.out_channels, self.in_channels, self.kernel_size)
            .contiguous()
        )
        if not input.is_complex():
            input = input.to(torch.cfloat)
        return F.conv1d(input, weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}"
        )


class MorletConv1d(ComplexGaborConv1d):
    r"""
    Learnable Complex Morlet Filterbank (1-D)
    -----------------------------------------

    Like :class:`ComplexGaborConv1d` but each atom is made **zero-mean** (the
    Morlet admissibility correction), subtracting the envelope-weighted mean of
    the carrier so the filter has no DC response.
    """

    def _kernels(self) -> torch.Tensor:
        env, carrier = self._env_carrier()
        base = env * carrier
        mean = base.sum(-1, keepdim=True) / env.sum(-1, keepdim=True)
        return base - env * mean
