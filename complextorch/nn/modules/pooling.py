from typing import Union, Tuple

import torch.nn as nn

from ... import CVTensor
from .. import functional as cvF

__all__ = ["CVAdaptiveAvgPool1d", "CVAdaptiveAvgPool2d", "CVAdaptiveAvgPool3d"]


class CVAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    r"""
    1-D Complex-Valued Adaptive Average Pooling
    -------------------------------------------
    
    Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool1d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html>`_ to the real and imaginary parts of the input tensor separately.
    
    Implements the following operation:
    
    .. math::
    
        G(\mathbf{z}) = \texttt{AdaptiveAvgPool1d}(\mathbf{z}_{real}) + j \texttt{AdaptiveAvgPool1d}(\mathbf{z}_{imag})
    """
    def __init__(self, output_size: Union[int, Tuple[int]]) -> None:
        super().__init__(output_size)

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool1d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html>`_ to the real and imaginary parts of the input tensor separately.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\texttt{AdaptiveAvgPool1d}(\mathbf{z}_{real}) + j \texttt{AdaptiveAvgPool1d}(\mathbf{z}_{imag})`
        """
        return cvF.apply_complex_split(super().forward, super().forward, input)


class CVAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    r"""
    2-D Complex-Valued Adaptive Average Pooling
    -------------------------------------------
    
    Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool2d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html>`_ to the real and imaginary parts of the input tensor separately.
    
    Implements the following operation:
    
    .. math::
    
        G(\mathbf{z}) = \texttt{AdaptiveAvgPool2d}(\mathbf{z}_{real}) + j \texttt{AdaptiveAvgPool2d}(\mathbf{z}_{imag})
    """
    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool2d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html>`_ to the real and imaginary parts of the input tensor separately.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\texttt{AdaptiveAvgPool2d}(\mathbf{z}_{real}) + j \texttt{AdaptiveAvgPool2d}(\mathbf{z}_{imag})`
        """
        return cvF.apply_complex_split(super().forward, super().forward, input)


class CVAdaptiveAvgPool3d(nn.AdaptiveAvgPool3d):
    r"""
    3-D Complex-Valued Adaptive Average Pooling
    -------------------------------------------
    
    Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool3d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html>`_ to the real and imaginary parts of the input tensor separately.
    
    Implements the following operation:
    
    .. math::
    
        G(\mathbf{z}) = \texttt{AdaptiveAvgPool3d}(\mathbf{z}_{real}) + j \texttt{AdaptiveAvgPool3d}(\mathbf{z}_{imag})
    """
    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool3d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html>`_ to the real and imaginary parts of the input tensor separately.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\texttt{AdaptiveAvgPool3d}(\mathbf{z}_{real}) + j \texttt{AdaptiveAvgPool3d}(\mathbf{z}_{imag})`
        """
        return cvF.apply_complex_split(super().forward, super().forward, input)
