r"""
Complex-Valued VD / ARD Linear & Bilinear Layers
================================================

Adapted from :mod:`cplxmodule.nn.relevance.complex.{base,vd,ard}` for native
``torch.cfloat``. See :class:`complextorch.nn.relevance.BaseARD` for the
shared interface.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from complextorch.nn.relevance._expi import torch_expi
from complextorch.nn.relevance.base import BaseARD
from complextorch.nn.utils.sparsity import SparsityStats

__all__ = ["LinearVD", "BilinearVD", "LinearARD", "BilinearARD"]


def _init_complex_weight(weight: torch.Tensor, in_features: int) -> None:
    bound = 1.0 / math.sqrt(in_features)
    with torch.no_grad():
        weight.real.uniform_(-bound, bound)
        weight.imag.uniform_(-bound, bound)


def _init_complex_bias(bias: torch.Tensor, in_features: int) -> None:
    bound = 1.0 / math.sqrt(in_features)
    with torch.no_grad():
        bias.real.uniform_(-bound, bound)
        bias.imag.uniform_(-bound, bound)


class _GaussianMixin:
    r"""Provides ``log_alpha`` and a small ``reset_variational_parameters``."""

    def reset_variational_parameters(self) -> None:
        with torch.no_grad():
            self.log_sigma2.fill_(-10.0)

    @property
    def log_alpha(self) -> torch.Tensor:
        return self.log_sigma2 - 2.0 * torch.log(self.weight.abs() + 1e-12)


class _RelevanceMixin(SparsityStats):
    __sparsity_ignore__ = ("log_sigma2",)

    def relevance(self, *, threshold: float, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return (self.log_alpha <= threshold).to(self.log_alpha.dtype)

    def sparsity(self, *, threshold: float, **kwargs):
        relevance = self.relevance(threshold=threshold)
        n_dropped = float(self.weight.numel()) - float(relevance.sum().item())
        return [(id(self.weight), n_dropped)]


class _CplxVDMixin:
    r"""KL of complex Gaussian posterior vs scale-free log-uniform prior."""

    @property
    def penalty(self) -> torch.Tensor:
        n_log_alpha = -self.log_alpha
        # Euler-Mascheroni constant ensures non-negativity.
        return float(np.euler_gamma) + n_log_alpha - torch_expi(-torch.exp(n_log_alpha))


class _CplxARDMixin:
    r"""Empirical-Bayes (softplus) penalty."""

    @property
    def penalty(self) -> torch.Tensor:
        return F.softplus(-self.log_alpha)


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


class _LinearGaussian(_GaussianMixin, nn.Module):
    r"""Complex linear with multiplicative Gaussian noise on the weight."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.cfloat)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter("bias", None)
        self.log_sigma2 = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_complex_weight(self.weight, self.in_features)
        if self.bias is not None:
            _init_complex_bias(self.bias, self.in_features)
        self.reset_variational_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu = F.linear(input, self.weight, self.bias)
        if not self.training:
            return mu
        # Variance of the additive noise: linear(|x|^2, exp(log_sigma^2)).
        s2 = F.linear(
            input.real * input.real + input.imag * input.imag,
            torch.exp(self.log_sigma2),
            None,
        )
        s2 = s2.clamp_min(1e-8)
        # Circular complex N(0, s2) noise.
        eps_r = torch.randn_like(s2)
        eps_i = torch.randn_like(s2)
        noise = torch.complex(eps_r, eps_i) * (torch.sqrt(s2 / 2.0))
        return mu + noise


class LinearVD(_CplxVDMixin, _RelevanceMixin, _LinearGaussian, BaseARD):
    r"""Complex Linear with Variational Dropout (log-uniform prior, exact KL via Ei)."""

    pass


class LinearARD(_CplxARDMixin, _RelevanceMixin, _LinearGaussian, BaseARD):
    r"""Complex Linear with Automatic Relevance Determination (softplus penalty)."""

    pass


# ---------------------------------------------------------------------------
# Bilinear
# ---------------------------------------------------------------------------


class _BilinearGaussian(_GaussianMixin, nn.Module):
    r"""Complex bilinear with multiplicative Gaussian noise on the weight."""

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        conjugate: bool = True,
    ) -> None:
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.conjugate = conjugate
        self.weight = nn.Parameter(
            torch.empty(out_features, in1_features, in2_features, dtype=torch.cfloat)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter("bias", None)
        self.log_sigma2 = nn.Parameter(
            torch.empty(out_features, in1_features, in2_features)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in1_features)
        with torch.no_grad():
            self.weight.real.uniform_(-bound, bound)
            self.weight.imag.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.real.uniform_(-bound, bound)
                self.bias.imag.uniform_(-bound, bound)
        self.reset_variational_parameters()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        x1 = input1.conj() if self.conjugate else input1
        mu = torch.einsum("...i,kij,...j->...k", x1, self.weight, input2)
        if self.bias is not None:
            mu = mu + self.bias
        if not self.training:
            return mu
        # Variance per output: einsum(|x1|^2, exp(log_sigma2), |x2|^2)
        m1 = input1.real * input1.real + input1.imag * input1.imag
        m2 = input2.real * input2.real + input2.imag * input2.imag
        s2 = torch.einsum("...i,kij,...j->...k", m1, torch.exp(self.log_sigma2), m2)
        s2 = s2.clamp_min(1e-8)
        eps_r = torch.randn_like(s2)
        eps_i = torch.randn_like(s2)
        noise = torch.complex(eps_r, eps_i) * (torch.sqrt(s2 / 2.0))
        return mu + noise


class BilinearVD(_CplxVDMixin, _RelevanceMixin, _BilinearGaussian, BaseARD):
    pass


class BilinearARD(_CplxARDMixin, _RelevanceMixin, _BilinearGaussian, BaseARD):
    pass
