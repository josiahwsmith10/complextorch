from typing import Optional

import torch.nn as nn

from ... import CVTensor
from .. import functional as cvF

__all__ = ["CVSoftmax", "MagSoftmax", "MagMinMaxNorm"]


class CVSoftmax(nn.Module):
    """
    Split Complex-Valued Softmax Layer
    ----------------------------------

    Simple real/image split softmax function.
    Applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ to the real and imaginary parts of the input tensor.

    Note: this naive implementation can cause significant phase changes and the relationship between the real and iamginary parts of the complex-valued signal are ignored in the two SoftMax computations.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \\texttt{SoftMax}(\mathbf{z}_{real}) + j \\texttt{SoftMax}(\mathbf{z}_{imag})
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super(CVSoftmax, self).__init__()

        self.softmax = nn.Softmax(dim)

    def forward(self, input: CVTensor) -> CVTensor:
        """Computes softmax over the real and imaginary parts of the input tensor separately

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\\texttt{SoftMax}(\mathbf{z}_{real}) + j \\texttt{SoftMax}(\mathbf{z}_{imag})`
        """
        return cvF.apply_complex_split(self.softmax, self.softmax, input)


class PhaseSoftmax(nn.Module):
    """
    Phase-Preserving Complex-Valued Softmax Layer
    ---------------------------------------------

    Retains phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ function to magnitude.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \\texttt{SoftMax}(|\mathbf{z}|) * \mathbf{z} / |\mathbf{z}|
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super(PhaseSoftmax, self).__init__()

        self.softmax = nn.Softmax(dim)

    def forward(self, input: CVTensor) -> CVTensor:
        """Retains phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ function to magnitude.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\\texttt{SoftMax}(|\mathbf{z}|) * \mathbf{z} / |\mathbf{z}|`
        """
        x_mag = input.abs()
        return self.softmax(x_mag) * (input / x_mag)


class MagSoftmax(nn.Module):
    """
    Magnitude Softmax Layer
    -----------------------

    Ignores phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ to the magnitude.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \\texttt{SoftMax}(|\mathbf{z}|)
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super(MagSoftmax, self).__init__()

        self.softmax = nn.Softmax(dim)

    def forward(self, input: CVTensor) -> CVTensor:
        """Ignores phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ function to magnitude.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\\texttt{SoftMax}(|\mathbf{z}|)`
        """
        return self.softmax(input.abs())


class MagMinMaxNorm(nn.Module):
    """
    Magnitude Min-Max Normalization Layer
    -------------------------------------

    Applies the *min-max norm* to the input tensor yielding an output whose magnitude is normalized between 0 and 1 over the specified dimension while phase information remains unchanged.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \\frac{\mathbf{z} - \mathbf{z}_{min}}{\mathbf{z}_{max} - \mathbf{z}_{min}}
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super(MagMinMaxNorm, self).__init__()

        self.dim = dim

    def forward(self, input: CVTensor) -> CVTensor:
        """Applies the *min-max norm* to the input tensor yielding an output whose magnitude is normalized between 0 and 1 over the specified dimension while phase information remains unchanged.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\\frac{\mathbf{z} - \mathbf{z}_{min}}{\mathbf{z}_{max} - \mathbf{z}_{min}}`
        """
        x_mag = input.abs()
        x_min = x_mag.min(self.dim, keepdim=True)[0]
        x_max = x_mag.max(self.dim, keepdim=True)[0]
        out = (input - x_min) / (x_max - x_min)
        return CVTensor(out.real, out.imag)
