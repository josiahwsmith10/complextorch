import torch
import torch.nn as nn
import torch.nn.functional as F

from complextorch.nn import functional as cvF

__all__ = ["Dropout", "Dropout1d", "Dropout2d", "Dropout3d"]


class Dropout(nn.Module):
    r"""
    Complex-Valued Dropout Layer
    ----------------------------

    Applies `PyTorch Dropout <https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html>`_ to real and imaginary parts separately, with **independent** Bernoulli masks per part.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{Dropout}(\mathbf{x}) + j \texttt{Dropout}(\mathbf{y}),

    where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`.

    .. note::

        This differs from the dropout used in *Trabelsi et al. (2018) Deep Complex
        Networks*, which uses a **shared** Bernoulli mask so that the entire complex
        value is dropped together (preserving the phase of the surviving entries).
        Because the real and imaginary masks here are sampled independently, the
        phase of a non-dropped entry can change when only one of its real/imag
        parts is zeroed out. Choose this layer deliberately.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(Dropout, self).__init__()

        self.dropout_r = nn.Dropout(p, inplace)
        self.dropout_i = nn.Dropout(p, inplace)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Performs complex-valued dropout on the input tensor

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\texttt{Dropout}(\mathbf{x}) + j \texttt{Dropout}(\mathbf{y})`
        """
        return cvF.apply_complex_split(self.dropout_r, self.dropout_i, input)


class _ChannelDropoutNd(nn.Module):
    r"""Internal base for channel-wise complex dropout with a shared real/imag mask.

    Implements Trabelsi et al. (2018) "Deep Complex Networks" complex dropout:
    one Bernoulli mask is drawn per channel and applied to **both** the real and
    imaginary parts simultaneously, so an entire complex channel is dropped
    together and the phase of surviving entries is preserved.
    """

    _dropout_fn = staticmethod(F.dropout1d)  # overridden in subclasses

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return input
        if not input.is_complex():
            return self._dropout_fn(input, self.p, training=True, inplace=self.inplace)
        # view_as_real returns a real view of shape (..., 2); the channel-wise
        # dropout treats element 0 of that final dim as the "channel" axis, so
        # we drop both real and imag together by operating on view_as_real then
        # going back to complex.
        real_view = torch.view_as_real(input)
        # Move the trailing size-2 axis to a leading position so dropoutNd's
        # channel semantics see ``channels`` as the leading dim of the original
        # input, not the {re,im} axis. Easiest: build a fake-channel tensor of
        # shape (B, C, ...) and apply mask manually.
        # Sample a Bernoulli mask of shape (B, C, 1, 1, ...) matching the input's
        # spatial rank, then scale by 1/(1-p).
        b, c = input.shape[0], input.shape[1]
        mask_shape = (b, c) + (1,) * (input.dim() - 2)
        mask = torch.empty(mask_shape, dtype=input.real.dtype, device=input.device)
        mask.bernoulli_(1 - self.p).div_(1 - self.p)
        return input * mask

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


class Dropout1d(_ChannelDropoutNd):
    r"""
    Complex-Valued 1-D Channel Dropout (Trabelsi 2018 shared mask)
    --------------------------------------------------------------

    Zeros out entire complex channels (matching :class:`torch.nn.Dropout1d`)
    using a single Bernoulli mask per channel applied to both real and
    imaginary parts. The phase of surviving entries is preserved.

    Input shape: ``(B, C, L)``.
    """

    _dropout_fn = staticmethod(F.dropout1d)


class Dropout2d(_ChannelDropoutNd):
    r"""
    Complex-Valued 2-D Channel Dropout (Trabelsi 2018 shared mask)
    --------------------------------------------------------------

    Zeros out entire complex channels (matching :class:`torch.nn.Dropout2d`)
    using a single Bernoulli mask per channel applied to both real and
    imaginary parts.

    Input shape: ``(B, C, H, W)``.
    """

    _dropout_fn = staticmethod(F.dropout2d)


class Dropout3d(_ChannelDropoutNd):
    r"""
    Complex-Valued 3-D Channel Dropout (Trabelsi 2018 shared mask)
    --------------------------------------------------------------

    Zeros out entire complex channels (matching :class:`torch.nn.Dropout3d`)
    using a single Bernoulli mask per channel applied to both real and
    imaginary parts.

    Input shape: ``(B, C, D, H, W)``.
    """

    _dropout_fn = staticmethod(F.dropout3d)
