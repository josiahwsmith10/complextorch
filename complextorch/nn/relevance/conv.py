r"""
Complex-Valued VD / ARD Conv1d/2d/3d Layers
===========================================

Adapted from :mod:`cplxmodule.nn.relevance.complex` for native ``torch.cfloat``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from complextorch.nn.relevance.base import BaseARD
from complextorch.nn.relevance.linear import (
    _CplxARDMixin,
    _CplxVDMixin,
    _GaussianMixin,
    _RelevanceMixin,
)

__all__ = [
    "Conv1dARD",
    "Conv1dVD",
    "Conv2dARD",
    "Conv2dVD",
    "Conv3dARD",
    "Conv3dVD",
]


def _to_tuple(x, n: int) -> tuple[int, ...]:
    if isinstance(x, int):
        return (x,) * n
    return tuple(x)


class _ConvNdGaussian(_GaussianMixin, nn.Module):
    """Internal base — subclasses set ``_conv_fn`` and ``_nd``."""

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
    ) -> None:
        super().__init__()
        if padding_mode != "zeros":
            raise ValueError(
                f"Only padding_mode='zeros' is supported, got {padding_mode!r}"
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
        self.weight = nn.Parameter(torch.empty(*weight_shape, dtype=torch.cfloat))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, dtype=torch.cfloat))
        else:
            self.register_parameter("bias", None)
        self.log_sigma2 = nn.Parameter(torch.empty(*weight_shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.in_channels // self.groups
        for k in self.kernel_size:
            fan_in *= k
        bound = 1.0 / math.sqrt(fan_in)
        with torch.no_grad():
            self.weight.real.uniform_(-bound, bound)
            self.weight.imag.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.real.uniform_(-bound, bound)
                self.bias.imag.uniform_(-bound, bound)
        self.reset_variational_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu = self._conv_fn(
            input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if not self.training:
            return mu
        s2 = self._conv_fn(
            input.real * input.real + input.imag * input.imag,
            torch.exp(self.log_sigma2),
            None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        s2 = s2.clamp_min(1e-8)
        eps_r = torch.randn_like(s2)
        eps_i = torch.randn_like(s2)
        noise = torch.complex(eps_r, eps_i) * (torch.sqrt(s2 / 2.0))
        return mu + noise


class _Conv1dGaussian(_ConvNdGaussian):
    _conv_fn = staticmethod(F.conv1d)
    _nd = 1


class _Conv2dGaussian(_ConvNdGaussian):
    _conv_fn = staticmethod(F.conv2d)
    _nd = 2


class _Conv3dGaussian(_ConvNdGaussian):
    _conv_fn = staticmethod(F.conv3d)
    _nd = 3


class Conv1dVD(_CplxVDMixin, _RelevanceMixin, _Conv1dGaussian, BaseARD):
    pass


class Conv2dVD(_CplxVDMixin, _RelevanceMixin, _Conv2dGaussian, BaseARD):
    pass


class Conv3dVD(_CplxVDMixin, _RelevanceMixin, _Conv3dGaussian, BaseARD):
    pass


class Conv1dARD(_CplxARDMixin, _RelevanceMixin, _Conv1dGaussian, BaseARD):
    pass


class Conv2dARD(_CplxARDMixin, _RelevanceMixin, _Conv2dGaussian, BaseARD):
    pass


class Conv3dARD(_CplxARDMixin, _RelevanceMixin, _Conv3dGaussian, BaseARD):
    pass
