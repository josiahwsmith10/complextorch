import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Linear"]


class Linear(nn.Module):
    r"""
    Gauss-trick Complex-Valued Linear Layer
    ---------------------------------------

    Real/imag-split implementation of complex linear multiplication using
    Gauss' multiplication trick (3 real matmuls instead of the naive 4).
    Mirrors :class:`torch.nn.Linear`, analogous to the Gauss-trick
    convolutions :class:`complextorch.nn.gauss.Conv1d` and siblings.

    For ordinary use prefer :class:`complextorch.nn.Linear`, which wraps
    ``torch.nn.Linear`` with ``dtype=torch.cfloat`` and is faster on modern
    PyTorch.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

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
