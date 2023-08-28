import torch.nn as nn

from ... import CVTensor

__all__ = ["ComplexRatioMask"]


class ComplexRatioMask(nn.Module):
    """
    Complex Ratio Mask.

    G(x) = sigmoid(|x|) * x / |x|

    Retains phase and squeezes magnitude using sigmoid function.
    """

    def __init__(self) -> None:
        super(ComplexRatioMask, self).__init__()

    def forward(self, input: CVTensor) -> CVTensor:
        x_mag = input.abs()
        return x_mag.sigmoid() * (input / x_mag)
