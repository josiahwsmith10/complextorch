import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import CVTensor

__all__ = ["CVLinear"]


class CVLinear(nn.Module):
    """
    Complex-Valued Linear Layer
    ---------------------------
    
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
        super(CVLinear, self).__init__()

        # Assumes PyTorch complex weight initialization is correct
        __temp = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=torch.complex64,
        )

        self.linear_r = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.linear_i = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.linear_r.weight.data = __temp.weight.real
        self.linear_i.weight.data = __temp.weight.imag

        if bias:
            self.linear_r.bias.data = __temp.bias.real
            self.linear_i.bias.data = __temp.bias.imag

    @property
    def weight(self) -> CVTensor:
        return CVTensor(self.linear_r.weight, self.linear_i.weight)

    @property
    def bias(self) -> CVTensor:
        if self.linear_r.bias is None:
            return None
        else:
            return CVTensor(self.linear_r.bias, self.linear_i.bias)

    def forward(self, input: CVTensor) -> CVTensor:
        """
        Computes multiplication 25% faster than naive method by using Gauss' multiplication trick
        """
        t1 = self.linear_r(input.real)
        t2 = self.linear_i(input.imag)
        bias = (
            None
            if self.linear_r.bias is None
            else (self.linear_r.bias + self.linear_i.bias)
        )
        t3 = F.linear(
            input=(input.real + input.imag),
            weight=(self.linear_r.weight + self.linear_i.weight),
            bias=bias,
        )
        return CVTensor(t1 - t2, t3 - t2 - t1)
