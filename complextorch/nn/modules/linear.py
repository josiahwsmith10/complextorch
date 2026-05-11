import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Linear", "SlowLinear", "Bilinear"]


class Linear(nn.Module):
    r"""
    Complex-Valued Linear using PyTorch
    -----------------------------------

        - Implemented using `torch.nn.Linear <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_ and complex-valued tensors.

        - Used to be slower than `complextorch` version but is now faster after PyTorch 2.1.0 update.

        - Convenience wrapper over ``torch.nn.Linear`` whose only behavioural difference is the default ``dtype=torch.cfloat``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=torch.cfloat,
    ) -> None:
        super(Linear, self).__init__()

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes complex-valued convolution using PyTorch.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: Linear(input)
        """
        return self.linear(input)


class SlowLinear(nn.Module):
    r"""
    Slow Complex-Valued Linear Layer
    --------------------------------

    Follows `PyTorch implementation <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_ using Gauss' trick to improve the computation as in :doc:`Complex-Valued Convolution <./conv>`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(SlowLinear, self).__init__()

        # Assumes PyTorch complex weight initialization is correct
        __temp = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=torch.cfloat,
        )

        self.linear_r = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.linear_i = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.linear_r.weight.data = __temp.weight.real
        self.linear_i.weight.data = __temp.weight.imag

        if bias:
            self.bias_r = nn.Parameter(__temp.bias.real.detach().clone())
            self.bias_i = nn.Parameter(__temp.bias.imag.detach().clone())
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

    @property
    def weight(self) -> torch.Tensor:
        return torch.complex(self.linear_r.weight, self.linear_i.weight)

    @property
    def bias(self) -> torch.Tensor:
        if self.bias_r is None:
            return None
        return torch.complex(self.bias_r, self.bias_i)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes multiplication 25% faster than naive method by using Gauss' multiplication trick
        """
        t1 = self.linear_r(input.real)
        t2 = self.linear_i(input.imag)
        t3 = F.linear(
            input=(input.real + input.imag),
            weight=(self.linear_r.weight + self.linear_i.weight),
            bias=None,
        )
        out_r = t1 - t2
        out_i = t3 - t2 - t1
        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i
        return torch.complex(out_r, out_i)


class Bilinear(nn.Module):
    r"""
    Complex-Valued Bilinear Layer
    -----------------------------

    Applies a complex-valued bilinear transformation:

    .. math::

        y_k = \mathbf{x}_1^\dagger \mathbf{W}_k \mathbf{x}_2 + b_k \qquad (\text{conjugate=True, Hermitian})

        y_k = \mathbf{x}_1^\top    \mathbf{W}_k \mathbf{x}_2 + b_k \qquad (\text{conjugate=False, plain bilinear})

    With ``conjugate=True`` (default) the input ``x_1`` is conjugated before the
    contraction, giving the mathematically standard Hermitian inner-product form.
    Setting ``conjugate=False`` uses a plain bilinear product, matching the
    real-valued :class:`torch.nn.Bilinear` semantics.

    Args:
        in1_features: size of the first input.
        in2_features: size of the second input.
        out_features: size of the output.
        bias: if ``True``, adds a learnable bias.
        conjugate: if ``True`` (default), uses Hermitian form.
        device, dtype: standard PyTorch factory kwargs; ``dtype`` defaults to
            ``torch.cfloat``.
    """

    __constants__ = ["in1_features", "in2_features", "out_features", "conjugate"]

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        conjugate: bool = True,
        device=None,
        dtype=torch.cfloat,
    ) -> None:
        super(Bilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.conjugate = conjugate

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in1_features, in2_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in1_features)
        with torch.no_grad():
            self.weight.real.uniform_(-bound, bound)
            self.weight.imag.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.real.uniform_(-bound, bound)
                self.bias.imag.uniform_(-bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        x1 = input1.conj() if self.conjugate else input1
        out = torch.einsum("...i,kij,...j->...k", x1, self.weight, input2)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}, "
            f"conjugate={self.conjugate}"
        )
