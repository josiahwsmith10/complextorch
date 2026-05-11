r"""
Masked Linear / Bilinear (Complex)
==================================

Layers that apply a fixed binary mask to their complex weight at forward
time. Used to deploy a learned-sparsity pattern at inference.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from complextorch.nn.masked.base import BaseMasked, MaskedWeightMixin

__all__ = ["LinearMasked", "BilinearMasked"]


def _init_complex_weight(weight: torch.Tensor, fan_in: int) -> None:
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        weight.real.uniform_(-bound, bound)
        weight.imag.uniform_(-bound, bound)


class LinearMasked(MaskedWeightMixin, BaseMasked):
    r"""Complex linear with a fixed binary weight mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype: torch.dtype = torch.cfloat,
    ) -> None:
        super().__init__()  # BaseMasked.__init__ -> registers `mask` buffer
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        _init_complex_weight(self.weight, in_features)
        if self.bias is not None:
            _init_complex_weight(self.bias, in_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight_masked if self.is_sparse else self.weight
        return F.linear(input, w, self.bias)


class BilinearMasked(MaskedWeightMixin, BaseMasked):
    r"""Complex bilinear with a fixed binary weight mask."""

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        conjugate: bool = True,
        device=None,
        dtype: torch.dtype = torch.cfloat,
    ) -> None:
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.conjugate = conjugate
        self.weight = nn.Parameter(
            torch.empty(
                out_features, in1_features, in2_features, device=device, dtype=dtype
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        _init_complex_weight(self.weight, in1_features)
        if self.bias is not None:
            _init_complex_weight(self.bias, in1_features)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        x1 = input1.conj() if self.conjugate else input1
        w = self.weight_masked if self.is_sparse else self.weight
        out = torch.einsum("...i,kij,...j->...k", x1, w, input2)
        if self.bias is not None:
            out = out + self.bias
        return out
