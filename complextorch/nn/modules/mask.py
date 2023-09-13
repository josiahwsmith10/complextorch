from typing import Optional

import torch.nn as nn

from ... import CVTensor

__all__ = ["ComplexRatioMask", "MagMinMaxNorm"]


class ComplexRatioMask(nn.Module):
    r"""
    Complex Ratio Mask
    ------------------

    .. math::

        G(\mathbf{z}) = \text{sigmoid}(|\mathbf{z}|) * \mathbf{z} / |\mathbf{z}|

    Retains phase and squeezes magnitude using `sigmoid function <https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html>`_.

    Based on work from the following paper:

        **HW Cho, S Choi, YR Cho, and J Kim: Complex-Valued Channel Attention and Application in Ego-Velocity Estimation With Automotive Radar**

            - See [23]

            - https://ieeexplore.ieee.org/abstract/document/9335579
    """

    def __init__(self) -> None:
        super(ComplexRatioMask, self).__init__()

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Computes complex ratio mask on complex-valued input tensor.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\text{sigmoid}(|\mathbf{z}|) * \mathbf{z} / |\mathbf{z}|`
        """
        x_mag = input.abs()
        return x_mag.sigmoid() * (input / x_mag)


class MagMinMaxNorm(nn.Module):
    r"""
    Magnitude Min-Max Normalization Layer
    -------------------------------------

    Applies the *min-max norm* to the input tensor yielding an output whose magnitude is normalized between 0 and 1 over the specified dimension while phase information remains unchanged.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \frac{\mathbf{z} - \mathbf{z}_{min}}{\mathbf{z}_{max} - \mathbf{z}_{min}}
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super(MagMinMaxNorm, self).__init__()

        self.dim = dim

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Applies the *min-max norm* to the input tensor yielding an output whose magnitude is normalized between 0 and 1 over the specified dimension while phase information remains unchanged.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\frac{\mathbf{z} - \mathbf{z}_{min}}{\mathbf{z}_{max} - \mathbf{z}_{min}}`
        """
        x_mag = input.abs()
        x_min = x_mag.min(self.dim, keepdim=True)[0]
        x_max = x_mag.max(self.dim, keepdim=True)[0]
        out = (input - x_min) / (x_max - x_min)
        return CVTensor(out.real, out.imag)
