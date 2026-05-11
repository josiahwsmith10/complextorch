r"""
Torch-Native Complex Transforms
===============================

Class-based transforms operating on torch tensors. All transforms are
:class:`torch.nn.Module` subclasses so they compose inside
:class:`torch.nn.Sequential` (or :class:`torchvision.transforms.Compose`).
"""

import math

import torch
import torch.nn as nn

from complextorch.transforms.functional import (
    _applyfft2,
    _applyifft2,
    _center_crop,
    _check_chw,
    _log_normalize_amplitude,
    _padifneeded,
    _spatial_resize_bicubic,
)

__all__ = [
    "FFT2",
    "HWC2CHW",
    "IFFT2",
    "Amplitude",
    "CenterCrop",
    "FFTResize",
    "LogAmplitude",
    "Normalize",
    "PadIfNeeded",
    "PolSAR",
    "RandomPhase",
    "RealImaginary",
    "SpatialResize",
    "ToImaginary",
    "ToReal",
    "ToTensor",
    "Unsqueeze",
]


# ---------------------------------------------------------------------------
# I/O and shape transforms
# ---------------------------------------------------------------------------


class ToTensor(nn.Module):
    r"""Cast input to a :class:`torch.Tensor` of the requested ``dtype``.

    Args:
        dtype: target dtype. Defaults to ``torch.cfloat`` since most
            :mod:`complextorch` workflows expect complex inputs.
    """

    def __init__(self, dtype: torch.dtype = torch.cfloat) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, x) -> torch.Tensor:
        t = torch.as_tensor(x)
        return t.to(self.dtype)

    def extra_repr(self) -> str:
        return f"dtype={self.dtype}"


class Unsqueeze(nn.Module):
    r"""Insert a size-1 dimension at ``dim``.

    Thin :class:`torch.nn.Module` wrapper around :meth:`torch.Tensor.unsqueeze`,
    intended for use inside a :class:`torchvision.transforms.Compose` pipeline.

    Args:
        dim: position at which the new axis is inserted.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class HWC2CHW(nn.Module):
    r"""Permute ``(H, W, C)`` to ``(C, H, W)``.

    PIL / NumPy image conventions store channels-last; PyTorch expects
    channels-first. Raises :class:`ValueError` on inputs that are not 3-D.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"HWC2CHW expects a 3-D tensor, got {tuple(x.shape)}")
        return x.permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# Magnitude / component extraction
# ---------------------------------------------------------------------------


class LogAmplitude(nn.Module):
    r"""
    ``log1p(|x| / scale) * exp(j*arg x)`` (or real magnitude if ``preserve_phase=False``).

    Standard SAR preprocessing: raw SAR magnitudes span many orders of
    magnitude, so the log-scaling makes them tractable for a network.
    """

    def __init__(self, scale: float = 1.0, preserve_phase: bool = True) -> None:
        super().__init__()
        self.scale = scale
        self.preserve_phase = preserve_phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _log_normalize_amplitude(x, self.scale, self.preserve_phase)


class Amplitude(nn.Module):
    r"""Returns ``|x|`` (complex -> real)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs()


class ToReal(nn.Module):
    r"""Return :math:`\Re(x)`, i.e. the real part of a complex tensor.

    No-op for inputs that are already real (returned unchanged), so the
    transform is safe to use unconditionally in a preprocessing pipeline.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.real if x.is_complex() else x


class ToImaginary(nn.Module):
    r"""Return :math:`\Im(x)`, i.e. the imaginary part of a complex tensor.

    For real-valued inputs, returns a tensor of zeros with the same shape
    and dtype so the transform composes cleanly with mixed pipelines.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.imag if x.is_complex() else torch.zeros_like(x)


class RealImaginary(nn.Module):
    r"""
    Stack real and imaginary parts along the channel dim.

    Complex ``(C, H, W)`` -> real ``(2C, H, W)``; ``(B, C, H, W)`` ->
    ``(B, 2C, H, W)``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            return x
        return torch.cat([x.real, x.imag], dim=-3)


# ---------------------------------------------------------------------------
# Statistics / randomization
# ---------------------------------------------------------------------------


class Normalize(nn.Module):
    r"""
    Per-channel 2x2 whitening with precomputed statistics.

    Given per-channel ``mean`` (complex, shape ``(C,)``) and ``covariance``
    (real, shape ``(C, 2, 2)``), applies
    ``(x - mean) @ cov^{-1/2}`` per channel. The 2x2 matrix square root is
    computed via :func:`complextorch.nn.functional.inv_sqrtm2x2`.
    """

    def __init__(self, mean: torch.Tensor, covariance: torch.Tensor) -> None:
        super().__init__()
        if covariance.shape[-2:] != (2, 2):
            raise ValueError(
                f"covariance must have shape (..., 2, 2), got {tuple(covariance.shape)}"
            )
        self.register_buffer("mean", mean.to(torch.cfloat))
        self.register_buffer("covariance", covariance.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from complextorch.nn.functional import inv_sqrtm2x2

        # Contract: (C, H, W) or (B, C, H, W) — channel at dim -3, two spatial dims.
        if x.dim() not in (3, 4) or x.shape[-3] != self.mean.shape[0]:
            raise ValueError(
                f"expected input of shape (C, H, W) or (B, C, H, W) with C={self.mean.shape[0]}, "
                f"got shape {tuple(x.shape)}"
            )
        m = self.mean.view(self.mean.shape[0], 1, 1)
        x = x - m
        a = self.covariance[..., 0, 0].view(-1, 1, 1)
        b = self.covariance[..., 0, 1].view(-1, 1, 1)
        c = self.covariance[..., 1, 0].view(-1, 1, 1)
        d = self.covariance[..., 1, 1].view(-1, 1, 1)
        w, xc, yc, z = inv_sqrtm2x2(a, b, c, d)
        re = w * x.real + xc * x.imag
        im = yc * x.real + z * x.imag
        return torch.complex(re, im)


class RandomPhase(nn.Module):
    r"""Multiply by ``exp(j * phi)`` with ``phi ~ Uniform(0, 2*pi)`` (or ``[-pi, pi]`` if ``centered``).

    Phase-invariance data augmentation for coherent signals.
    """

    def __init__(self, centered: bool = False) -> None:
        super().__init__()
        self.centered = centered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.centered:
            phi = torch.empty((), device=x.device).uniform_(-math.pi, math.pi)
        else:
            phi = torch.empty((), device=x.device).uniform_(0.0, 2.0 * math.pi)
        rotor = torch.polar(torch.tensor(1.0, device=x.device), phi)
        if x.is_complex():
            return x * rotor
        return x.to(torch.cfloat) * rotor


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------


class PadIfNeeded(nn.Module):
    r"""Symmetric padding to bring ``(H, W)`` up to at least ``(min_h, min_w)``.

    ``mode`` matches :func:`torch.nn.functional.pad`: ``'constant'``,
    ``'reflect'``, ``'replicate'``, ``'circular'``.
    """

    def __init__(self, min_h: int, min_w: int, mode: str = "constant") -> None:
        super().__init__()
        self.min_h = min_h
        self.min_w = min_w
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_chw(x)
        return _padifneeded(x, self.min_h, self.min_w, mode=self.mode)


class CenterCrop(nn.Module):
    r"""Center-crop to ``(h, w)``."""

    def __init__(self, h: int, w: int) -> None:
        super().__init__()
        self.h = h
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _center_crop(x, self.h, self.w)


class SpatialResize(nn.Module):
    r"""Bicubic resize to ``(h, w)`` (split real/imag for complex inputs)."""

    def __init__(self, h: int, w: int) -> None:
        super().__init__()
        self.h = h
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _spatial_resize_bicubic(x, self.h, self.w)


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------


class FFT2(nn.Module):
    r"""2-D FFT with zero-frequency centering (``fftshift(fft2(x))``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _applyfft2(x)


class IFFT2(nn.Module):
    r"""Inverse of :class:`FFT2`: ``ifft2(ifftshift(x))``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _applyifft2(x)


class FFTResize(nn.Module):
    r"""
    Spectral-domain resize.

    FFT -> centre-crop or zero-pad the spectrum to ``(h, w)`` -> inverse FFT.
    Preserves spectral characteristics of coherent signals (useful for SAR);
    with ``energy_preserving=True``, scales the spectrum so the total power
    matches the spatial-resize convention.
    """

    def __init__(self, h: int, w: int, energy_preserving: bool = True) -> None:
        super().__init__()
        self.h = h
        self.w = w
        self.energy_preserving = energy_preserving

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = _applyfft2(x)
        H, W = spec.shape[-2], spec.shape[-1]
        # Crop or zero-pad to (h, w).
        spec = _resize_spectrum(spec, self.h, self.w)
        if self.energy_preserving:
            scale = math.sqrt((self.h * self.w) / (H * W))
            spec = spec * scale
        return _applyifft2(spec)


def _resize_spectrum(spec: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Centre-crop or zero-pad the trailing (H, W) of the complex ``spec`` to ``(h, w)``."""
    H, W = spec.shape[-2], spec.shape[-1]
    # Vertical
    if h <= H:
        top = (H - h) // 2
        spec = spec[..., top : top + h, :]
    else:
        pad_top = (h - H) // 2
        pad_bot = h - H - pad_top
        pad_r = torch.nn.functional.pad(spec.real, (0, 0, pad_top, pad_bot))
        pad_i = torch.nn.functional.pad(spec.imag, (0, 0, pad_top, pad_bot))
        spec = torch.complex(pad_r, pad_i)
    # Horizontal
    if w <= W:
        left = (W - w) // 2
        spec = spec[..., :, left : left + w]
    else:
        pad_l = (w - W) // 2
        pad_r = w - W - pad_l
        pad_re = torch.nn.functional.pad(spec.real, (pad_l, pad_r, 0, 0))
        pad_im = torch.nn.functional.pad(spec.imag, (pad_l, pad_r, 0, 0))
        spec = torch.complex(pad_re, pad_im)
    return spec


# ---------------------------------------------------------------------------
# Polarimetric SAR
# ---------------------------------------------------------------------------


class PolSAR(nn.Module):
    r"""
    PolSAR channel selection.

    Input is a complex tensor with ``C`` channels following the standard
    quad-pol order (HH, HV, VH, VV) — at minimum the first ``C`` of those.
    Reduces to ``out_channels`` channels per:

    - C=1: identity (any out_channels=1).
    - C=2 (HH, VV by convention here, matching torchcvnn): out=1 -> [HH];
      out=2 -> [HH, VV].
    - C=3 (HH, VV, HV by convention): out=1 -> [HH]; out=2 -> [HH, VV];
      out=3 -> all.
    - C=4: out=1 -> [HH]; out=2 -> [HH, VV]; out=3 -> [HH, VV, (HV+VH)/2];
      out=4 -> all.
    """

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        if out_channels < 1 or out_channels > 4:
            raise ValueError(f"out_channels must be in [1, 4], got {out_channels}")
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(f"expected at least 3 dims, got {tuple(x.shape)}")
        c = x.shape[-3]
        out = self.out_channels
        if c == 1:
            if out != 1:
                raise ValueError(
                    f"single-channel input requires out_channels=1, got {out}"
                )
            return x
        if c == 2:
            if out == 1:
                return x[..., 0:1, :, :]
            if out == 2:
                return x
            raise ValueError(
                f"2-channel input supports out_channels in (1, 2), got {out}"
            )
        if c == 3:
            if out == 1:
                return x[..., 0:1, :, :]
            if out == 2:
                return x[..., :2, :, :]
            if out == 3:
                return x
            raise ValueError(
                f"3-channel input supports out_channels in (1, 2, 3), got {out}"
            )
        if c == 4:
            hh, hv, vh, vv = (
                x[..., 0, :, :],
                x[..., 1, :, :],
                x[..., 2, :, :],
                x[..., 3, :, :],
            )
            if out == 1:
                return hh.unsqueeze(-3)
            if out == 2:
                return torch.stack([hh, vv], dim=-3)
            if out == 3:
                return torch.stack([hh, vv, 0.5 * (hv + vh)], dim=-3)
            if out == 4:
                return x
        raise ValueError(f"unsupported input channel count {c}")
