import torch
import torch.nn as nn

from .... import CVTensor

__all__ = ["CVSigmoid", "modReLU", "zReLU", "CVCardiod", "CVSigLog"]


class CVSigmoid(nn.Module):
    """
    Complex-Valued Sigmoid Activation Function.

    T Nitta, Y Kuroe. Hyperbolic Gradient Operator and Hyperbolic Back-Propagation Learning Algorithms.
    Eq. (71)
    https://ieeexplore.ieee.org/abstract/document/7885584
    """

    def __init__(self) -> None:
        super(CVSigmoid, self).__init__()

    def forward(self, input: CVTensor) -> CVTensor:
        out = 1 / (1 + torch.exp(input.complex))
        return CVTensor(out.real, out.imag)


class modReLU(nn.Module):
    """
    modulus Rectified Linear Unit.

    modReLU(z) = ReLU(|z| + b) * z / |z| = ReLU(|z| + b) * exp(1j * angle(z))

    Notice that |z| (z.abs()) is always positive, so if b > 0  then |z| + b > = 0 always.
    In order to have any non-linearity effect, b must be smaller than 0 (b<0).

    Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks.
    Eq. (8)
    https://arxiv.org/abs/1511.06464


    J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.
    Eq. (36)
    https://arxiv.org/abs/2302.08286
    """

    def __init__(self, bias: float = 0.0) -> None:
        super(modReLU, self).__init__()

        assert bias < 0, "bias must be smaller than 0 to have a non-linearity effect"

        self.relu = nn.ReLU()
        self.bias = bias

    def forward(self, input: CVTensor) -> CVTensor:
        out = self.relu(input.abs() + self.bias) * torch.exp(1j * input.angle())
        return CVTensor(out.real, out.imag)


class zReLU(nn.Module):
    """
    Guberman ReLU.

    Nitzan Guberman. On complex valued convolutional neural networks.
    Section 4.2.1
    https://arxiv.org/abs/1602.09046

    Deep Complex Networks.
    Eq. (5)
    https://arxiv.org/abs/1705.09792


    J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.
    Eq. (35)
    https://arxiv.org/abs/2302.08286
    """

    def __init__(self) -> None:
        super(zReLU, self).__init__()

    def forward(self, input: CVTensor) -> CVTensor:
        x_angle = input.angle()
        mask = (0 <= x_angle) & (x_angle <= torch.pi / 2)
        out = input * mask
        return CVTensor(out.real, out.imag)


class CVCardiod(nn.Module):
    """
    Cardiod Activation Function.

    CVCardiod(z) = 1/2 * (1 + cos(angle(z))) * z

    Patrick Virtue, Stella X. Yu, Michael Lustig. Better than Real: Complex-valued Neural Nets for MRI Fingerprinting.
    Eq. (3)
    https://arxiv.org/abs/1707.00070


    J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.
    Eq. (37)
    https://arxiv.org/abs/2302.08286
    """

    def __init__(self) -> None:
        super(CVCardiod, self).__init__()

    def forward(self, input: CVTensor) -> CVTensor:
        return 0.5 * (1 + torch.cos(input.angle())) * input


class CVSigLog(nn.Module):
    """
    CVSigLog Activation Function.

    SigLog(z) = z / (c + 1/r * |z|)

    G.M. Georgiou and C. Koutsougeras. Complex domain backpropagation.
    Eq. (20)
    https://ieeexplore.ieee.org/abstract/document/142037
    """

    def __init__(self, c: float = 1.0, r: float = 1.0) -> None:
        super(CVSigLog, self).__init__()

        self.c = c
        self.r = r

    def forward(self, input: CVTensor) -> CVTensor:
        return input / (self.c + input.abs() / self.r)
