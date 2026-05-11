r"""
Layout-Conversion Modules
=========================

Drop-in :class:`torch.nn.Module` adapters that convert between real-tensor
layouts and complex tensors, so they compose inside :class:`torch.nn.Sequential`.

These layouts come up in real-to-complex pipelines:

- **Interleaved**: ``[..., 2*D]`` with real and imaginary parts interleaved:
  ``(re_0, im_0, re_1, im_1, ...)``.
- **Concatenated**: ``[..., 2*D]`` with all real parts followed by all
  imaginary parts: ``(re_0, ..., re_{D-1}, im_0, ..., im_{D-1})``.

If your data is in ``(..., 2)`` final-dim layout (one slot for real, one for
imag), use :func:`torch.view_as_complex` / :func:`torch.view_as_real` directly
— no wrapper is needed.
"""

import torch
import torch.nn as nn

__all__ = [
    "InterleavedToComplex",
    "ComplexToInterleaved",
    "ConcatenatedToComplex",
    "ComplexToConcatenated",
    "RealToComplex",
]


class InterleavedToComplex(nn.Module):
    r"""
    Interleaved Real Layout → Complex
    ---------------------------------

    Maps ``[..., 2D]`` with real/imag interleaved along the last dim to a
    complex tensor of shape ``[..., D]``.

    Input: ``(re_0, im_0, re_1, im_1, ...)``.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] % 2 != 0:
            raise ValueError(
                f"InterleavedToComplex requires last dim to be even; got {input.shape[-1]}"
            )
        reshaped = input.reshape(*input.shape[:-1], -1, 2).contiguous()
        return torch.view_as_complex(reshaped)


class ComplexToInterleaved(nn.Module):
    r"""
    Complex → Interleaved Real Layout
    ---------------------------------

    Inverse of :class:`InterleavedToComplex`. Maps complex ``[..., D]`` to real
    ``[..., 2D]`` with real/imag interleaved.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            raise TypeError(
                f"ComplexToInterleaved expects a complex input, got {input.dtype}"
            )
        as_real = torch.view_as_real(input)  # [..., D, 2]
        return as_real.reshape(*as_real.shape[:-2], -1)


class ConcatenatedToComplex(nn.Module):
    r"""
    Concatenated Real Layout → Complex
    ----------------------------------

    Maps ``[..., 2D]`` with the first ``D`` slots being real parts and the last
    ``D`` slots being imaginary parts to a complex tensor of shape ``[..., D]``.

    Input: ``(re_0, ..., re_{D-1}, im_0, ..., im_{D-1})``.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] % 2 != 0:
            raise ValueError(
                f"ConcatenatedToComplex requires last dim to be even; got {input.shape[-1]}"
            )
        real, imag = torch.chunk(input, 2, dim=-1)
        return torch.complex(real.contiguous(), imag.contiguous())


class ComplexToConcatenated(nn.Module):
    r"""
    Complex → Concatenated Real Layout
    ----------------------------------

    Inverse of :class:`ConcatenatedToComplex`. Maps complex ``[..., D]`` to
    real ``[..., 2D]`` with all real parts followed by all imaginary parts.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            raise TypeError(
                f"ComplexToConcatenated expects a complex input, got {input.dtype}"
            )
        return torch.cat((input.real, input.imag), dim=-1)


class RealToComplex(nn.Module):
    r"""
    Real → Complex (Zero Imaginary)
    -------------------------------

    Lifts a real tensor into a complex tensor by setting the imaginary part to
    zero. Useful as the first layer of a network whose input is a real signal
    but whose internal representations are complex.
    """

    def __init__(self, dtype: torch.dtype = torch.cfloat) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_complex():
            return input.to(self.dtype)
        zeros = torch.zeros_like(input)
        return torch.complex(input, zeros).to(self.dtype)

    def extra_repr(self) -> str:
        return f"dtype={self.dtype}"
