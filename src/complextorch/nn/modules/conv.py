import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from typing import Tuple, Union

from .. import functional as cvF
from ... import CVTensor

__all__ = [
    "SlowCVConv1d",
    "CVConv1d",
    "CVConv2d",
    "CVConv3d",
    "CVConvTranpose1d",
    "CVConvTranpose2d",
    "CVConvTranpose3d",
    "default_slow_cvconv1d",
    "default_cvconv1d",
]


class SlowCVConv1d(nn.Module):
    """
    Slow complex-valued 1D convolution.
        - Implemented using torch.nn.Conv1d and complex-valued tensors.
        - slower than using CVTensor. PyTorch must have some additional overhead that makes
          this method significantly slower than using CVTensors and the other CVConv layers
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
        bias: bool = False,
    ) -> None:
        super(SlowCVConv1d, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            dtype=torch.complex64,
        )

    def forward(self, x: CVTensor) -> CVTensor:
        x = self.conv(x.complex)
        return CVTensor(x.real, x.imag)


def default_slow_cvconv1d(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = False
) -> SlowCVConv1d:
    """
    Default complex-valued 1D convolution.
        - Implemented using torch.nn.Conv1d and complex-valued tensors.
        - slower than using CVTensor
    """
    return SlowCVConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class _CVConv(nn.Module):
    """CVTensor-based complex-valued convolution."""

    def __init__(
        self,
        ConvClass: nn.Module,
        ConvFunc,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        super(_CVConv, self).__init__()
        
        self.ConvFunc = ConvFunc
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Assumes PyTorch complex weight initialization is correct
        __temp = ConvClass(
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
            dtype=torch.complex64,
        )

        self.conv_r = ConvClass(
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
        self.conv_i = ConvClass(
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

        self.conv_r.weight.data = __temp.weight.real
        self.conv_i.weight.data = __temp.weight.imag

        if bias:
            self.conv_r.bias.data = __temp.bias.real
            self.conv_i.bias.data = __temp.bias.imag

    @property
    def weight(self) -> CVTensor:
        return CVTensor(self.conv_r.weight, self.conv_i.weight)

    @property
    def bias(self) -> CVTensor:
        return CVTensor(self.conv_r.bias, self.conv_i.bias)

    def forward(self, x: CVTensor) -> CVTensor:
        """
        Computes convolution 25% faster than naive method by using Gauss'
        multiplication trick
        """
        t1 = self.conv_r(x.real)
        t2 = self.conv_i(x.imag)
        bias = None if self.conv_r.bias is None else (self.conv_r.bias + self.conv_i.bias)
        t3 = self.ConvFunc(
            input=(x.real + x.imag), 
            weight=(self.conv_r.weight + self.conv_i.weight), 
            bias=bias, 
            stride=self.stride, padding=self.padding, groups=self.groups
        )
        return CVTensor(t1 - t2, t3 - t2 - t1)


class CVConv1d(_CVConv):
    """CVTensor-based complex-valued 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super(CVConv1d, self).__init__(
            ConvClass=nn.Conv1d,
            ConvFunc=F.conv1d,
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


def default_cvconv1d(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = False
) -> CVConv1d:
    """Default complex-valued 1D convolution."""
    return CVConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class CVConv2d(_CVConv):
    """CVTensor-based complex-valued 2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super(CVConv2d, self).__init__(
            ConvClass=nn.Conv2d,
            ConvFunc=F.conv2d,
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


class CVConv3d(_CVConv):
    """CVTensor-based complex-valued 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super(CVConv3d, self).__init__(
            ConvClass=nn.Conv3d,
            ConvFunc=F.conv3d,
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


class _CVConvTranspose(nn.Module):
    """CVTensor-based complex-valued convolution."""

    def __init__(
        self,
        ConvClass,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        super(_CVConvTranspose, self).__init__()

        # Assumes PyTorch complex weight initialization is correct
        __temp = ConvClass(
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
            dtype=torch.complex64,
        )

        self.convt_r = ConvClass(
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
        self.convt_i = ConvClass(
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

        self.convt_r.weight.data = __temp.weight.real
        self.convt_i.weight.data = __temp.weight.imag

        if bias:
            self.convt_r.bias.data = __temp.bias.real
            self.convt_i.bias.data = __temp.bias.imag

    @property
    def weight(self) -> CVTensor:
        return CVTensor(self.convt_r.weight, self.convt_i.weight)

    @property
    def bias(self) -> CVTensor:
        return CVTensor(self.convt_r.bias, self.convt_i.bias)

    def forward(self, x: CVTensor) -> CVTensor:
        return cvF.apply_complex(self.convt_r, self.convt_i, x)


class CVConvTranpose1d(_CVConvTranspose):
    """CVTensor-based complex-valued 2D transpose convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super(CVConvTranpose1d, self).__init__(
            ConvClass=nn.ConvTranspose1d,
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


class CVConvTranpose2d(_CVConvTranspose):
    """CVTensor-based complex-valued 2D transpose convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super(CVConvTranpose2d, self).__init__(
            ConvClass=nn.ConvTranspose2d,
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


class CVConvTranpose3d(_CVConvTranspose):
    """CVTensor-based complex-valued 3D transpose convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super(CVConvTranpose3d, self).__init__(
            ConvClass=nn.ConvTranspose3d,
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
