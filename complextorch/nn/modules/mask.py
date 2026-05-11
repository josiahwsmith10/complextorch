from typing import Optional

import torch.nn as nn
import torch

__all__ = ["ComplexRatioMask", "PhaseSigmoid", "MagMinMaxNorm"]

_EPS = 1e-12


class ComplexRatioMask(nn.Module):
    r"""
    Complex Ratio Mask
    ------------------

    .. math::

        \texttt{ComplexRatioMask}(\mathbf{z}) = \texttt{Sigmoid}(|\mathbf{z}|) \odot \frac{\mathbf{z}}{|\mathbf{z}|}

    Retains phase and squeezes magnitude using `sigmoid function <https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html>`_.

    Based on work from the following paper:

        **HW Cho, S Choi, YR Cho, and J Kim: Complex-Valued Channel Attention and Application in Ego-Velocity Estimation With Automotive Radar**

            - See [23]

            - https://ieeexplore.ieee.org/abstract/document/9335579
    """

    def __init__(self) -> None:
        super(ComplexRatioMask, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes complex ratio mask on complex-valued input tensor.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\text{sigmoid}(|\mathbf{z}|) * \mathbf{z} / |\mathbf{z}|`
        """
        x_mag = input.abs()
        return x_mag.sigmoid() * (input / x_mag.clamp(min=_EPS))


class PhaseSigmoid(nn.Module):
    r"""
    Phase-Preserving Complex-Valued Sigmoid Layer
    ---------------------------------------------

    .. math::

        \texttt{PhaseSigmoid}(\mathbf{z}) = \texttt{Sigmoid}(|\mathbf{z}|) \odot \frac{\mathbf{z}}{|\mathbf{z}|}

    Retains phase and squeezes magnitude using `sigmoid function <https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html>`_.

    Based on work from the following paper:

        **HW Cho, S Choi, YR Cho, and J Kim: Complex-Valued Channel Attention and Application in Ego-Velocity Estimation With Automotive Radar**

            - See [23]

            - https://ieeexplore.ieee.org/abstract/document/9335579
    """

    def __init__(self) -> None:
        super(PhaseSigmoid, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes phase-preserving sigmoid mask on a complex-valued input tensor.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\text{sigmoid}(|\mathbf{z}|) * \mathbf{z} / |\mathbf{z}|`
        """
        x_mag = input.abs()
        return x_mag.sigmoid() * (input / x_mag.clamp(min=_EPS))


class MagMinMaxNorm(nn.Module):
    r"""
    Magnitude Min-Max Normalization Layer
    -------------------------------------

    Applies the *min-max norm* to the magnitude of the input tensor, yielding an
    output whose magnitude is normalized between 0 and 1 (over the specified
    dimension, if any) while phase information remains unchanged.

    Implements the following operation:

    .. math::

        \texttt{MagMinMaxNorm}(\mathbf{z}) = \frac{|\mathbf{z}| - |\mathbf{z}|_{min}}{|\mathbf{z}|_{max} - |\mathbf{z}|_{min}} \odot \exp(j \angle\mathbf{z})
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super(MagMinMaxNorm, self).__init__()

        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Applies the *min-max norm* to the magnitude of the input tensor while
        preserving phase.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: phase-preserving min-max normalized tensor
        """
        x_mag = input.abs()
        if self.dim is None:
            x_min = x_mag.min()
            x_max = x_mag.max()
        else:
            x_min = x_mag.min(dim=self.dim, keepdim=True).values
            x_max = x_mag.max(dim=self.dim, keepdim=True).values
        new_mag = (x_mag - x_min) / (x_max - x_min).clamp(min=_EPS)
        return torch.polar(new_mag, input.angle())
