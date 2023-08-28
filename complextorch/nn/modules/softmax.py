from typing import Optional

import torch.nn as nn

from ... import CVTensor
from .. import functional as cvF

__all__ = ["CVSoftmax", "MagSoftmax", "MagMinMaxNorm"]


class CVSoftmax(nn.Module):
    """Split Complex-Valued Softmax Layer."""

    def __init__(self, dim: Optional[int] = None) -> None:
        super(CVSoftmax, self).__init__()

        self.softmax_r = nn.Softmax(dim)
        self.softmax_i = nn.Softmax(dim)

    def forward(self, x: CVTensor) -> CVTensor:
        return cvF.apply_complex_split(self.softmax_r, self.softmax_i, x)


class PhaseSoftmax(nn.Module):
    """
    Phase-Preserving Complex-Valued Softmax Layer.

    G(x) = softmax(|x|) * x / |x|

    Retains phase and applies softmax function to magnitude.
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super(PhaseSoftmax, self).__init__()

        self.softmax = nn.Softmax(dim)

    def forward(self, x: CVTensor) -> CVTensor:
        x_mag = x.abs()
        return self.softmax(x_mag) * (x / x_mag)


class MagSoftmax(nn.Module):
    """Magnitude Softmax Layer."""

    def __init__(self, dim: Optional[int] = None) -> None:
        super(MagSoftmax, self).__init__()

        self.softmax = nn.Softmax(dim)

    def forward(self, x: CVTensor) -> CVTensor:
        return self.softmax(x.abs())


class MagMinMaxNorm(nn.Module):
    """Magnitude Min-Max Normalization Layer."""

    def __init__(self, dim: Optional[int] = None) -> None:
        super(MagMinMaxNorm, self).__init__()

        self.dim = dim

    def forward(self, x: CVTensor) -> CVTensor:
        x_mag = x.abs()
        x_min = x_mag.min(self.dim, keepdim=True)[0]
        x_max = x_mag.max(self.dim, keepdim=True)[0]
        out = (x - x_min) / (x_max - x_min)
        return CVTensor(out.real, out.imag)
