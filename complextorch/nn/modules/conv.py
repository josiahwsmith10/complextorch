import torch
import torch.nn as nn

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]


class Conv1d(nn.Module):
    r"""
    Complex-Valued 1-D Convolution using PyTorch
    --------------------------------------------

        - Implemented using `torch.nn.Conv1d <https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html>`_ and complex-valued tensors.

        - Convenience wrapper over ``torch.nn.Conv1d`` whose only behavioural difference is the default ``dtype=torch.cfloat``.

        - See :mod:`complextorch.nn.gauss` for the hand-rolled real/imag-split variant using Gauss' trick.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=torch.cfloat,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes 1-D complex-valued convolution using PyTorch.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: Conv1d(input)
        """
        return self.conv(input)


class Conv2d(nn.Module):
    r"""
    Complex-Valued 2-D Convolution using PyTorch
    --------------------------------------------

        - Implemented using `torch.nn.Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_ and complex-valued tensors.

        - Convenience wrapper over ``torch.nn.Conv2d`` whose only behavioural difference is the default ``dtype=torch.cfloat``.

        - See :mod:`complextorch.nn.gauss` for the hand-rolled real/imag-split variant using Gauss' trick.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=torch.cfloat,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes 2-D complex-valued convolution using PyTorch.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: Conv2d(input)
        """
        return self.conv(input)


class Conv3d(nn.Module):
    r"""
    Complex-Valued 3-D Convolution using PyTorch
    --------------------------------------------

        - Implemented using `torch.nn.Conv3d <https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html>`_ and complex-valued tensors.

        - Convenience wrapper over ``torch.nn.Conv3d`` whose only behavioural difference is the default ``dtype=torch.cfloat``.

        - See :mod:`complextorch.nn.gauss` for the hand-rolled real/imag-split variant using Gauss' trick.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=torch.cfloat,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes 3-D complex-valued convolution using PyTorch.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: Conv3d(input)
        """
        return self.conv(input)


class ConvTranspose1d(nn.Module):
    r"""
    Complex-Valued 1-D Transposed Convolution using PyTorch
    -------------------------------------------------------

        - Implemented using `torch.nn.ConvTranspose1d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html>`_ and complex-valued tensors.

        - Convenience wrapper over ``torch.nn.ConvTranspose1d`` whose only behavioural difference is the default ``dtype=torch.cfloat``.

        - See :mod:`complextorch.nn.gauss` for the hand-rolled real/imag-split variant using Gauss' trick.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=torch.cfloat,
    ) -> None:
        super().__init__()

        self.conv_transposed = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes 1-D complex-valued transposed convolution using PyTorch.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: ConvTranspose1d(input)
        """
        return self.conv_transposed(input)


class ConvTranspose2d(nn.Module):
    r"""
    Complex-Valued 2-D Transposed Convolution using PyTorch
    -------------------------------------------------------

        - Implemented using `torch.nn.ConvTranspose2d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html>`_ and complex-valued tensors.

        - Convenience wrapper over ``torch.nn.ConvTranspose2d`` whose only behavioural difference is the default ``dtype=torch.cfloat``.

        - See :mod:`complextorch.nn.gauss` for the hand-rolled real/imag-split variant using Gauss' trick.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=torch.cfloat,
    ) -> None:
        super().__init__()

        self.conv_transposed = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes 2-D complex-valued transposed convolution using PyTorch.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: ConvTranspose2d(input)
        """
        return self.conv_transposed(input)


class ConvTranspose3d(nn.Module):
    r"""
    Complex-Valued 3-D Transposed Convolution using PyTorch
    -------------------------------------------------------

        - Implemented using `torch.nn.ConvTranspose3d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html>`_ and complex-valued tensors.

        - Convenience wrapper over ``torch.nn.ConvTranspose3d`` whose only behavioural difference is the default ``dtype=torch.cfloat``.

        - See :mod:`complextorch.nn.gauss` for the hand-rolled real/imag-split variant using Gauss' trick.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=torch.cfloat,
    ) -> None:
        super().__init__()

        self.conv_transposed = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes 3-D complex-valued transposed convolution using PyTorch.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: ConvTranspose3d(input)
        """
        return self.conv_transposed(input)
