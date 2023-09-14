import torch.nn as nn

from ... import CVTensor
from .. import functional as cvF

__all__ = ["CVDropout"]


class CVDropout(nn.Module):
    r"""
    Complex-Valued Dropout Layer
    ----------------------------

    Applies `PyTorch Droput <https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html>`_ to real and imaginary parts separately.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{Dropout}(\mathbf{x}) + j \texttt{Dropout}(\mathbf{y}),
        
    where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(CVDropout, self).__init__()

        self.dropout_r = nn.Dropout(p, inplace)
        self.dropout_i = nn.Dropout(p, inplace)

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Performs complex-valued dropout on the input tensor

        Args:
            x (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\texttt{Dropout}(\mathbf{x}) + j \texttt{Dropout}(\mathbf{y})`
        """
        return cvF.apply_complex_split(self.dropout_r, self.dropout_i, input)
