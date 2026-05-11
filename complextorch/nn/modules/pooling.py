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
    "SpectralPool1d",
    "SpectralPool2d",
    "SpectralPool3d",
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


def _spectral_pool(
    input: torch.Tensor,
    output_size: tuple[int, ...],
    spatial_dims: tuple[int, ...],
) -> torch.Tensor:
    r"""Downsample ``input`` along ``spatial_dims`` by cropping its centered DFT.

    Uses ``norm="forward"`` on both the forward and inverse FFT so that the
    spatial mean (i.e. the DC bin) is preserved exactly: if ``y`` is the
    output, ``y.mean(spatial_dims) == input.mean(spatial_dims)``.

    The crop is centered so that DC lands at position ``K // 2`` of the
    cropped spectrum, which is where ``torch.fft.ifftshift`` of a length-``K``
    signal expects it (this matters when ``K`` and the input length have
    different parities).
    """
    input_size = tuple(input.size(d) for d in spatial_dims)
    if any(k <= 0 for k in output_size):
        raise ValueError(f"output_size {output_size} must be positive")
    if any(k > n for k, n in zip(output_size, input_size, strict=True)):
        raise ValueError(
            f"output_size {output_size} must not exceed input spatial size "
            f"{input_size} along dims {spatial_dims}"
        )
    if output_size == input_size:
        return input
    z = torch.fft.fftn(input, dim=spatial_dims, norm="forward")
    z = torch.fft.fftshift(z, dim=spatial_dims)
    slices = [slice(None)] * input.dim()
    for d, n, k in zip(spatial_dims, input_size, output_size, strict=True):
        start = (n // 2) - (k // 2)
        slices[d] = slice(start, start + k)
    z = z[tuple(slices)]
    z = torch.fft.ifftshift(z, dim=spatial_dims)
    return torch.fft.ifftn(z, dim=spatial_dims, norm="forward")


def _to_tuple(value, n: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * n
    out = tuple(value)
    if len(out) != n:
        raise ValueError(f"{name} must have length {n}, got {len(out)}")
    return out


class _SpectralPoolNd(nn.Module):
    r"""Internal base for N-D spectral pooling. See :class:`SpectralPool1d`."""

    _ndim: int = 1  # overridden per dim

    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = _to_tuple(output_size, self._ndim, "output_size")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        spatial_dims = tuple(range(-self._ndim, 0))
        was_real = not input.is_complex()
        out = _spectral_pool(input, self.output_size, spatial_dims)
        if was_real:
            out = out.real
        return out

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class SpectralPool1d(_SpectralPoolNd):
    r"""
    1-D Spectral Pooling
    --------------------

    Downsamples the last spatial dimension by truncating the centered discrete
    Fourier spectrum, then inverting. Compared to average pooling, spectral
    pooling preserves more information per coefficient because it keeps the
    low-frequency content of the signal directly rather than averaging
    neighbouring samples.

    The forward and inverse FFTs use ``norm="forward"``, so the DC bin (and
    therefore the spatial mean) is preserved exactly. The crop is centered
    around DC.

    For real inputs the output is real; for complex inputs the output is
    complex. Only downsampling (``output_size <= input_size``) is supported.

    Based on:

        **O. Rippel, J. Snoek, R. P. Adams. Spectral Representations for
        Convolutional Neural Networks.** NeurIPS 2015.

            - https://arxiv.org/abs/1506.03767

    Also used as a complex-valued pooling layer in:

        **C. Trabelsi et al. Deep Complex Networks.** ICLR 2018.

            - https://arxiv.org/abs/1705.09792

    Args:
        output_size (int): number of spatial samples after pooling. Must be
            ``<= input.size(-1)``.
    """

    _ndim = 1


class SpectralPool2d(_SpectralPoolNd):
    r"""
    2-D Spectral Pooling
    --------------------

    See :class:`SpectralPool1d`. Operates on the last two spatial dimensions.

    Args:
        output_size (int or tuple[int, int]): spatial output shape ``(H, W)``.
            A single ``int`` is broadcast to both dimensions.
    """

    _ndim = 2


class SpectralPool3d(_SpectralPoolNd):
    r"""
    3-D Spectral Pooling
    --------------------

    See :class:`SpectralPool1d`. Operates on the last three spatial dimensions.

    Args:
        output_size (int or tuple[int, int, int]): spatial output shape
            ``(D, H, W)``. A single ``int`` is broadcast to all three
            dimensions.
    """

    _ndim = 3
