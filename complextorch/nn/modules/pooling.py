from typing import Union, Tuple

import torch.nn as nn
import torch

from .. import functional as cvF

__all__ = ["AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d"]


class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    r"""
    1-D Complex-Valued Adaptive Average Pooling
    -------------------------------------------

    Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool1d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html>`_ to the real and imaginary parts of the input tensor separately.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{AdaptiveAvgPool1d}(\mathbf{x}) + j \texttt{AdaptiveAvgPool1d}(\mathbf{y}),

    where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`
    """

    def __init__(self, output_size: Union[int, Tuple[int]]) -> None:
        super().__init__(output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool1d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html>`_ to the real and imaginary parts of the input tensor separately.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\texttt{AdaptiveAvgPool1d}(\mathbf{x}) + j \texttt{AdaptiveAvgPool1d}(\mathbf{y})`
        """
        return cvF.apply_complex_split(super().forward, super().forward, input)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    r"""
    2-D Complex-Valued Adaptive Average Pooling
    -------------------------------------------

    Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool2d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html>`_ to the real and imaginary parts of the input tensor separately.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{AdaptiveAvgPool2d}(\mathbf{x}) + j \texttt{AdaptiveAvgPool2d}(\mathbf{y}),

    where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`
    """

    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool2d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html>`_ to the real and imaginary parts of the input tensor separately.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\texttt{AdaptiveAvgPool2d}(\mathbf{x}) + j \texttt{AdaptiveAvgPool2d}(\mathbf{y})`
        """
        return cvF.apply_complex_split(super().forward, super().forward, input)


class AdaptiveAvgPool3d(nn.AdaptiveAvgPool3d):
    r"""
    3-D Complex-Valued Adaptive Average Pooling
    -------------------------------------------

    Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool3d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html>`_ to the real and imaginary parts of the input tensor separately.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{AdaptiveAvgPool3d}(\mathbf{x}) + j \texttt{AdaptiveAvgPool3d}(\mathbf{y}),

    where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`
    """

    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Applies adaptive average pooling using `torch.nn.AdaptiveAvgPool3d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html>`_ to the real and imaginary parts of the input tensor separately.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\texttt{AdaptiveAvgPool3d}(\mathbf{x}) + j \texttt{AdaptiveAvgPool3d}(\mathbf{y})`
        """
        return cvF.apply_complex_split(super().forward, super().forward, input)
