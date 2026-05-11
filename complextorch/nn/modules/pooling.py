import torch
import torch.nn as nn
import torch.nn.functional as F

from complextorch.nn import functional as cvF

__all__ = [
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "MagMaxPool1d",
    "MagMaxPool2d",
    "MagMaxPool3d",
]


class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    r"""
    1-D Complex-Valued Adaptive Average Pooling
    -------------------------------------------

    Applies adaptive average pooling using :class:`torch.nn.AdaptiveAvgPool1d` to the real and imaginary parts of the input tensor separately.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{AdaptiveAvgPool1d}(\mathbf{x}) + j \texttt{AdaptiveAvgPool1d}(\mathbf{y}),

    where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`
    """

    def __init__(self, output_size: int | tuple[int]) -> None:
        super().__init__(output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Applies adaptive average pooling using :class:`torch.nn.AdaptiveAvgPool1d` to the real and imaginary parts of the input tensor separately.

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

    Applies adaptive average pooling using :class:`torch.nn.AdaptiveAvgPool2d` to the real and imaginary parts of the input tensor separately.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{AdaptiveAvgPool2d}(\mathbf{x}) + j \texttt{AdaptiveAvgPool2d}(\mathbf{y}),

    where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`
    """

    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Applies adaptive average pooling using :class:`torch.nn.AdaptiveAvgPool2d` to the real and imaginary parts of the input tensor separately.

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

    Applies adaptive average pooling using :class:`torch.nn.AdaptiveAvgPool3d` to the real and imaginary parts of the input tensor separately.

    Implements the following operation:

    .. math::

        G(\mathbf{z}) = \texttt{AdaptiveAvgPool3d}(\mathbf{x}) + j \texttt{AdaptiveAvgPool3d}(\mathbf{y}),

    where :math:`\mathbf{z} = \mathbf{x} + j\mathbf{y}`
    """

    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Applies adaptive average pooling using :class:`torch.nn.AdaptiveAvgPool3d` to the real and imaginary parts of the input tensor separately.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\texttt{AdaptiveAvgPool3d}(\mathbf{x}) + j \texttt{AdaptiveAvgPool3d}(\mathbf{y})`
        """
        return cvF.apply_complex_split(super().forward, super().forward, input)


class AvgPool1d(nn.AvgPool1d):
    r"""
    1-D Complex-Valued Average Pooling
    ----------------------------------

    Convenience wrapper over :class:`torch.nn.AvgPool1d` for complex inputs.
    Average pooling is linear, so applying ``torch.nn.AvgPool1d`` to a
    ``torch.cfloat`` tensor is mathematically equivalent to pooling real and
    imaginary parts independently and recombining.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_complex():
            return cvF.apply_complex_split(super().forward, super().forward, input)
        return super().forward(input)


class AvgPool2d(nn.AvgPool2d):
    r"""
    2-D Complex-Valued Average Pooling
    ----------------------------------

    Convenience wrapper over :class:`torch.nn.AvgPool2d` for complex inputs.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_complex():
            return cvF.apply_complex_split(super().forward, super().forward, input)
        return super().forward(input)


class AvgPool3d(nn.AvgPool3d):
    r"""
    3-D Complex-Valued Average Pooling
    ----------------------------------

    Convenience wrapper over :class:`torch.nn.AvgPool3d` for complex inputs.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_complex():
            return cvF.apply_complex_split(super().forward, super().forward, input)
        return super().forward(input)


def _gather_max_by_magnitude(
    input: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    r"""Gather the *original* complex samples at ``indices`` (argmax-of-magnitude positions).

    ``indices`` come from ``F.max_poolNd_with_indices(|input|)``: they are linear
    indices into the spatial dimensions (last N dims of the input). For each
    output position, we select the corresponding complex sample from
    ``input`` and return it, preserving the original phase.
    """
    # Flatten spatial dimensions of input, then gather along that flat dim.
    n_spatial = indices.dim() - 2  # batch, channel, then spatial
    flat_input = input.reshape(*input.shape[:-n_spatial], -1)
    flat_indices = indices.reshape(*indices.shape[:-n_spatial], -1)
    gathered = torch.gather(flat_input, dim=-1, index=flat_indices)
    return gathered.reshape(*indices.shape)


class _MagMaxPoolNd(nn.Module):
    r"""Internal base for magnitude-argmax complex max pooling."""

    _max_pool_with_indices = staticmethod(F.max_pool1d_with_indices)

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input: torch.Tensor):
        magnitude = input.abs() if input.is_complex() else input
        _, indices = self._max_pool_with_indices(
            magnitude,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
        )
        if input.is_complex():
            out = _gather_max_by_magnitude(input, indices)
        else:
            # Real input: gather is equivalent to the max value itself
            out = _gather_max_by_magnitude(input, indices)
        if self.return_indices:
            return out, indices
        return out

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"ceil_mode={self.ceil_mode}"
        )


class MagMaxPool1d(_MagMaxPoolNd):
    r"""
    1-D Complex-Valued Max Pooling by Magnitude
    -------------------------------------------

    Pools by selecting the input position with the **largest magnitude**
    :math:`|z|` within each window, and returns the **original complex sample**
    at that position (phase preserved).

    Because ``torch.nn.MaxPool1d`` is not defined for complex tensors (no total
    ordering on :math:`\mathbb{C}`), this layer is the canonical complex
    analogue. Signature matches :class:`torch.nn.MaxPool1d`.
    """

    _max_pool_with_indices = staticmethod(F.max_pool1d_with_indices)


class MagMaxPool2d(_MagMaxPoolNd):
    r"""
    2-D Complex-Valued Max Pooling by Magnitude
    -------------------------------------------

    See :class:`MagMaxPool1d`. Signature matches :class:`torch.nn.MaxPool2d`.
    """

    _max_pool_with_indices = staticmethod(F.max_pool2d_with_indices)


class MagMaxPool3d(_MagMaxPoolNd):
    r"""
    3-D Complex-Valued Max Pooling by Magnitude
    -------------------------------------------

    See :class:`MagMaxPool1d`. Signature matches :class:`torch.nn.MaxPool3d`.
    """

    _max_pool_with_indices = staticmethod(F.max_pool3d_with_indices)
