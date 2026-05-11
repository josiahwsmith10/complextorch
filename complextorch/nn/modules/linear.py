import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Linear", "SlowLinear"]


class Linear(nn.Module):
    r"""
    Complex-Valued Linear using PyTorch
    -----------------------------------

        - Implemented using `torch.nn.Linear <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_ and complex-valued tensors.

        - Used to be slower than `complextorch` version but is now faster after PyTorch 2.1.0 update.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
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
        bias: bool = False,
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
