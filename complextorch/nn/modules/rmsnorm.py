r"""
Complex-Valued RMSNorm
======================

Root-Mean-Square layer normalization adapted to complex tensors. Equivalent to
:class:`torch.nn.RMSNorm` but operates on complex inputs; the affine parameter
is a 2x2 real matrix applied to ``(Re, Im)`` of each channel.
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    r"""
    Complex-Valued RMS Normalization
    --------------------------------

    .. math::

        y = \frac{x}{\sqrt{\text{mean}(|x|^2) + \epsilon}}

    Followed by an optional per-feature affine transform: the real/imag pair
    of each feature is multiplied by a learnable 2x2 matrix. No bias.

    Args:
        normalized_shape: shape of the trailing dims to normalize over (same
            semantics as :class:`torch.nn.RMSNorm`).
        eps: numerical stabilizer.
        elementwise_affine: if ``True``, applies a learnable 2x2 affine.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Tuple[int, ...], torch.Size],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            # 2x2 affine on (Re, Im); initialized to identity / sqrt(2) so the
            # zero-mean unit-variance assumption maps to a proper complex
            # standardisation (same convention as the LayerNorm in this lib).
            self.weight = nn.Parameter(torch.empty(2, 2, *normalized_shape))
        else:
            self.register_parameter("weight", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.elementwise_affine:
            return
        eye = (0.70710678118 * torch.eye(2)).view(
            2, 2, *([1] * len(self.normalized_shape))
        )
        self.weight.data.copy_(eye)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            raise TypeError(f"RMSNorm expects a complex input, got dtype={input.dtype}")
        # Compute RMS of |x|^2 over the normalized_shape (last len(...) dims).
        ns = len(self.normalized_shape)
        dims = tuple(range(input.dim() - ns, input.dim()))
        rms = torch.sqrt(input.abs().pow(2).mean(dim=dims, keepdim=True) + self.eps)
        normed = input / rms
        if not self.elementwise_affine:
            return normed

        # Apply 2x2 affine: weight has shape (2, 2, *normalized_shape).
        re, im = normed.real, normed.imag
        w = self.weight
        out_r = w[0, 0] * re + w[0, 1] * im
        out_i = w[1, 0] * re + w[1, 1] * im
        return torch.complex(out_r, out_i)

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
