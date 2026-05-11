r"""
Complex-Valued Upsampling Modules
=================================

Two flavors of complex-valued upsampling / interpolation:

- :class:`Upsample` — *split form*. Interpolates the real and imaginary parts
  independently. Matches the behavior of :class:`torchcvnn.nn.Upsample` and
  ``complexPyTorch.complex_upsample``.

- :class:`PolarUpsample` — *polar form*. Interpolates the magnitude
  :math:`|z|` and the phase :math:`\arg z` independently, then recombines via
  :math:`|z| \cdot \exp(j\,\arg z)`. Phase-preserving along smooth phase
  regions; useful for coherent signal models (radar, SAR). The cost is a
  visible discontinuity wherever the phase wraps from :math:`-\pi` to
  :math:`+\pi` — neither form is universally correct.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Upsample", "PolarUpsample"]


_SizeT = Optional[Union[int, Tuple[int, ...]]]


def _interpolate(
    x: torch.Tensor,
    size: _SizeT,
    scale_factor: Optional[Union[float, Tuple[float, ...]]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
) -> torch.Tensor:
    return F.interpolate(
        x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )


class Upsample(nn.Module):
    r"""
    Complex-Valued Upsample (split form)
    ------------------------------------

    Applies :func:`torch.nn.functional.interpolate` independently to the real
    and imaginary parts of a complex input, then recombines.

    All keyword arguments mirror :class:`torch.nn.Upsample`.
    """

    def __init__(
        self,
        size: _SizeT = None,
        scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            return _interpolate(
                input,
                self.size,
                self.scale_factor,
                self.mode,
                self.align_corners,
                self.recompute_scale_factor,
            )
        real = _interpolate(
            input.real,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            self.recompute_scale_factor,
        )
        imag = _interpolate(
            input.imag,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            self.recompute_scale_factor,
        )
        return torch.complex(real, imag)

    def extra_repr(self) -> str:
        return (
            f"size={self.size}, scale_factor={self.scale_factor}, "
            f"mode={self.mode!r}, align_corners={self.align_corners}"
        )


class PolarUpsample(nn.Module):
    r"""
    Complex-Valued Upsample (polar form)
    ------------------------------------

    Interpolates magnitude and phase independently. For a complex input
    :math:`z = |z|\, e^{j\arg z}`:

    .. math::

        |z|' = \text{interp}(|z|), \quad
        \arg z' = \text{interp}(\arg z), \quad
        z' = |z|' \cdot e^{j\,\arg z'}

    Phase-preserving along smooth phase regions but introduces discontinuities
    at phase wraps (:math:`\pm\pi`). Choose between :class:`Upsample` (split)
    and :class:`PolarUpsample` (polar) based on whether your data has smooth
    phase (favor polar) or smooth real/imag parts (favor split).

    All keyword arguments mirror :class:`torch.nn.Upsample`.
    """

    def __init__(
        self,
        size: _SizeT = None,
        scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            return _interpolate(
                input,
                self.size,
                self.scale_factor,
                self.mode,
                self.align_corners,
                self.recompute_scale_factor,
            )
        mag = input.abs()
        phase = input.angle()
        mag_up = _interpolate(
            mag,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            self.recompute_scale_factor,
        )
        phase_up = _interpolate(
            phase,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            self.recompute_scale_factor,
        )
        return torch.polar(mag_up, phase_up)

    def extra_repr(self) -> str:
        return (
            f"size={self.size}, scale_factor={self.scale_factor}, "
            f"mode={self.mode!r}, align_corners={self.align_corners}"
        )
