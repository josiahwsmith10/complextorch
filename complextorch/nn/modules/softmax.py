from typing import Optional

import torch.nn as nn

from ... import CVTensor
from .. import functional as cvF

__all__ = ["CVSoftMax", "MagSoftMax", "PhaseSoftMax"]


class CVSoftMax(nn.Module):
    r"""
    Split Complex-Valued Softmax Layer
    ----------------------------------

    Simple real/image split softmax function.
    Applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ to the real and imaginary parts of the input tensor.

    Note: this naive implementation can cause significant phase changes and the relationship between the real and iamginary parts of the complex-valued signal are ignored in the two SoftMax computations.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{SoftMax}(\mathbf{x}) + j \texttt{SoftMax}(\mathbf{y}),
        
    where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super(CVSoftMax, self).__init__()

        self.softmax = nn.Softmax(dim)

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Computes softmax over the real and imaginary parts of the input tensor separately

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\texttt{SoftMax}(\mathbf{x}) + j \texttt{SoftMax}(\mathbf{y})`
        """
        return cvF.apply_complex_split(self.softmax, self.softmax, input)


class PhaseSoftMax(nn.Module):
    r"""
    Phase-Preserving Complex-Valued Softmax Layer
    ---------------------------------------------

    Retains phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ function to magnitude.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{SoftMax}(|\mathbf{z}|) \odot \mathbf{z} / |\mathbf{z}|
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super(PhaseSoftMax, self).__init__()

        self.softmax = nn.Softmax(dim)

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Retains phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ function to magnitude.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\texttt{SoftMax}(|\mathbf{z}|) \odot \mathbf{z} / |\mathbf{z}|`
        """
        x_mag = input.abs()
        return self.softmax(x_mag) * (input / x_mag)


class MagSoftMax(nn.Module):
    r"""
    Magnitude Softmax Layer
    -----------------------

    Ignores phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ to the magnitude.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{SoftMax}(|\mathbf{z}|)
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super(MagSoftMax, self).__init__()

        self.softmax = nn.Softmax(dim)

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Ignores phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ function to magnitude.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\texttt{SoftMax}(|\mathbf{z}|)`
        """
        return self.softmax(input.abs())
