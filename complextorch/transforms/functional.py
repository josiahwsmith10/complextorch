r"""
Functional Transform Helpers
============================

Lower-level helpers backing the class transforms in
:mod:`complextorch.transforms.transforms`. Most are intentionally private
(``_``-prefixed); the public ones are listed in ``__all__``.
"""

import torch
import torch.nn.functional as F

__all__ = ["polsar_dict_to_array", "rescale_intensity"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def polsar_dict_to_array(
    d: dict[str, torch.Tensor], order: tuple[str, ...] = ("HH", "HV", "VH", "VV")
) -> torch.Tensor:
    r"""
    Stack a polarimetric SAR channel dictionary into a tensor.

    Args:
        d: mapping from polarization name (e.g. ``'HH'``) to a 2-D complex
            tensor of shape ``(H, W)`` (or higher-rank; all entries must agree).
        order: which channels to keep, in the desired output order. Defaults
            to the standard quad-pol order ``(HH, HV, VH, VV)``.

    Returns:
        Complex tensor of shape ``(len(order), H, W)``.
    """
    chans = [d[k] for k in order if k in d]
    if not chans:
        raise ValueError(
            f"none of the requested channels {order} are present in dict keys {tuple(d.keys())}"
        )
    return torch.stack(chans, dim=0)


def rescale_intensity(
    x: torch.Tensor,
    in_range: tuple[float, float] | None = None,
    out_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    r"""
    Linearly remap intensity values from ``in_range`` to ``out_range``.

    Mirrors :func:`skimage.exposure.rescale_intensity` for torch tensors.
    Values outside ``in_range`` are clamped to the corresponding output
    bound. ``in_range=None`` uses ``(x.min(), x.max())``.
    """
    if x.is_complex():
        raise TypeError("rescale_intensity expects a real tensor; pass abs(x) first")
    if in_range is None:
        in_lo, in_hi = float(x.min()), float(x.max())
    else:
        in_lo, in_hi = in_range
    out_lo, out_hi = out_range
    if in_hi == in_lo:
        return torch.full_like(x, out_lo)
    x = x.clamp(in_lo, in_hi)
    return (x - in_lo) / (in_hi - in_lo) * (out_hi - out_lo) + out_lo


# ---------------------------------------------------------------------------
# Private helpers backing the class transforms
# ---------------------------------------------------------------------------


def _check_chw(x: torch.Tensor) -> None:
    if x.dim() not in (3, 4):
        raise ValueError(
            f"expected a (C, H, W) or (B, C, H, W) tensor, got {tuple(x.shape)}"
        )


def _log_normalize_amplitude(
    x: torch.Tensor, scale: float = 1.0, preserve_phase: bool = True
) -> torch.Tensor:
    """``log1p(|x| / scale) * exp(j * arg x)`` when ``preserve_phase``; else real magnitude."""
    out_mag = torch.log1p(x.abs() / scale)
    if preserve_phase and x.is_complex():
        return torch.polar(out_mag, x.angle())
    return out_mag


def _applyfft2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))


def _applyifft2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)))


def _get_padding(
    shape: tuple[int, int], target: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Symmetric (left, right, top, bottom) pad amounts to bring (H, W) up to target."""
    h, w = shape
    th, tw = target
    pad_h = max(th - h, 0)
    pad_w = max(tw - w, 0)
    top = pad_h // 2
    bot = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return left, right, top, bot


def _padifneeded(
    x: torch.Tensor, min_h: int, min_w: int, mode: str = "constant"
) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    if h >= min_h and w >= min_w:
        return x
    left, right, top, bot = _get_padding((h, w), (min_h, min_w))
    if x.is_complex():
        re = F.pad(x.real, (left, right, top, bot), mode=mode)
        im = F.pad(x.imag, (left, right, top, bot), mode=mode)
        return torch.complex(re, im)
    return F.pad(x, (left, right, top, bot), mode=mode)


def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    H, W = x.shape[-2], x.shape[-1]
    if h > H or w > W:
        raise ValueError(f"center_crop: target {(h, w)} larger than input {(H, W)}")
    top = (H - h) // 2
    left = (W - w) // 2
    return x[..., top : top + h, left : left + w]


def _spatial_resize_bicubic(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # F.interpolate(mode='bicubic') doesn't support complex; do split.
    needs_unsqueeze = x.dim() == 3
    if needs_unsqueeze:
        x = x.unsqueeze(0)
    if x.is_complex():
        re = F.interpolate(x.real, size=(h, w), mode="bicubic", align_corners=False)
        im = F.interpolate(x.imag, size=(h, w), mode="bicubic", align_corners=False)
        out = torch.complex(re, im)
    else:
        out = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False)
    return out.squeeze(0) if needs_unsqueeze else out
