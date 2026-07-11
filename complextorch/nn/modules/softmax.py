import torch
import torch.nn as nn

from complextorch.nn import functional as cvF

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

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim)

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Computes softmax over the real and imaginary parts of the input tensor separately

        Args:
            input (torch.Tensor): input tensor
            mask (torch.Tensor, optional): real additive mask broadcastable to
                ``input`` (``-inf`` disables a position). Added to **both**
                the real and imaginary parts, so a masked position gets
                exactly zero weight in each split softmax.

        Returns:
            torch.Tensor: :math:`\texttt{SoftMax}(\mathbf{x}) + j \texttt{SoftMax}(\mathbf{y})`
        """
        if mask is not None:
            input = torch.complex(input.real + mask, input.imag + mask)
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

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim)

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Retains phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ function to magnitude.

        Args:
            input (torch.Tensor): input tensor
            mask (torch.Tensor, optional): real additive mask broadcastable to
                ``input`` (``-inf`` disables a position). Added to the
                **magnitude logits** — adding ``-inf`` to the complex input
                itself would make :math:`|z| = +\infty` and hand the masked
                position all the weight instead of none.

        Returns:
            torch.Tensor: :math:`\texttt{SoftMax}(|\mathbf{z}|) \odot \mathbf{z} / |\mathbf{z}|`
        """
        x_mag = input.abs()
        logits = x_mag if mask is None else x_mag + mask
        return self.softmax(logits) * (input / x_mag.clamp(min=1e-12))


class MagSoftMax(nn.Module):
    r"""
    Magnitude Softmax Layer
    -----------------------

    Ignores phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ to the magnitude.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{SoftMax}(|\mathbf{z}|)
    """

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim)

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Ignores phase and applies `SoftMax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ function to magnitude.

        Args:
            input (torch.Tensor): input tensor
            mask (torch.Tensor, optional): real additive mask broadcastable to
                ``input`` (``-inf`` disables a position), added to the
                magnitude logits.

        Returns:
            torch.Tensor: :math:`\texttt{SoftMax}(|\mathbf{z}|)`
        """
        logits = input.abs() if mask is None else input.abs() + mask
        return self.softmax(logits)
