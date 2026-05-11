r"""
Masked Conv (Complex)
=====================
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from complextorch.nn.masked.base import BaseMasked, MaskedWeightMixin

__all__ = ["Conv1dMasked", "Conv2dMasked", "Conv3dMasked"]


def _to_tuple(x, n: int) -> tuple[int, ...]:
    if isinstance(x, int):
        return (x,) * n
    return tuple(x)


class _ConvMaskedNd(MaskedWeightMixin, BaseMasked):
    _conv_fn = staticmethod(F.conv1d)
    _nd = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype: torch.dtype = torch.cfloat,
    ) -> None:
        super().__init__()
        if padding_mode != "zeros":
            raise ValueError(
                f"Only padding_mode='zeros' is supported in masked conv, got {padding_mode!r}"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_tuple(kernel_size, self._nd)
        self.stride = _to_tuple(stride, self._nd)
        self.padding = (
            padding if isinstance(padding, str) else _to_tuple(padding, self._nd)
        )
        self.dilation = _to_tuple(dilation, self._nd)
        self.groups = groups

        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight = nn.Parameter(
            torch.empty(*weight_shape, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        fan_in = in_channels // groups
        for k in self.kernel_size:
            fan_in *= k
        bound = 1.0 / math.sqrt(fan_in)
        with torch.no_grad():
            self.weight.real.uniform_(-bound, bound)
            self.weight.imag.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.real.uniform_(-bound, bound)
                self.bias.imag.uniform_(-bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight_masked if self.is_sparse else self.weight
        return self._conv_fn(
            input,
            w,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv1dMasked(_ConvMaskedNd):
    _conv_fn = staticmethod(F.conv1d)
    _nd = 1


class Conv2dMasked(_ConvMaskedNd):
    _conv_fn = staticmethod(F.conv2d)
    _nd = 2


class Conv3dMasked(_ConvMaskedNd):
    _conv_fn = staticmethod(F.conv3d)
    _nd = 3
