import torch
import torch.nn as nn

from .... import CVTensor

__all__ = ["CVSigmoid", "zReLU", "CVCardiod", "CVSigLog"]


class CVSigmoid(nn.Module):
    r"""
    Complex-Valued Sigmoid Activation Function
    ------------------------------------------

    An extension of the sigmoid activation function to the complex domain.

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \frac{1}{1 + \exp{(\mathbf{z})}}.

    Based on work from the following paper:

        **T Nitta, Y Kuroe. Hyperbolic Gradient Operator and Hyperbolic Back-Propagation Learning Algorithms.**

            - Eq. (71)

            - https://ieeexplore.ieee.org/abstract/document/7885584
    """

    def __init__(self) -> None:
        super(CVSigmoid, self).__init__()

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Computes the complex-valued sigmoid activation function.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\frac{1}{1 + \exp{(\mathbf{z})}}`
        """
        out = 1 / (1 + torch.exp(input.complex))
        return CVTensor(out.real, out.imag)


class zReLU(nn.Module):
    r"""
    Guberman ReLU
    -------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \begin{cases} \mathbf{z} \quad \text{if} \quad \angle\mathbf{z} \in [0, \pi/2] \\ 0 \quad \text{else} \end{cases}

    Based on work from the following papers:

        **Nitzan Guberman. On complex valued convolutional neural networks.**

            - Section 4.2.1

            - https://arxiv.org/abs/1602.09046

        Deep Complex Networks.

            - Eq. (5)

            - https://arxiv.org/abs/1705.09792


        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Eq. (35)

            - https://arxiv.org/abs/2302.08286
    """

    def __init__(self) -> None:
        super(zReLU, self).__init__()

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Computes the complex-valued Guberman ReLU.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\begin{cases} \mathbf{z} \quad \text{if} \quad \angle\mathbf{z} \in [0, \pi/2] \\ 0 \quad \text{else} \end{cases}`
        """
        x_angle = input.angle()
        mask = (0 <= x_angle) & (x_angle <= torch.pi / 2)
        out = input * mask
        return CVTensor(out.real, out.imag)


class CVCardiod(nn.Module):
    r"""
    Cardiod Activation Function
    ---------------------------

    Implements the operation:

    .. math::

        G(z) = \frac{1}{2} (1 + \text{cos}(\angle\mathbf{z})) \odot \mathbf{z}

    Based on work from the following papers:

        **Patrick Virtue, Stella X. Yu, Michael Lustig. Better than Real: Complex-valued Neural Nets for MRI Fingerprinting.**

            - Eq. (3)

            - https://arxiv.org/abs/1707.00070


        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Eq. (37)

            - https://arxiv.org/abs/2302.08286
    """

    def __init__(self) -> None:
        super(CVCardiod, self).__init__()

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Computes the complex-valued cardioid activation function.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\frac{1}{2} (1 + \text{cos}(\angle\mathbf{z})) \odot \mathbf{z}`
        """
        return 0.5 * (1 + torch.cos(input.angle())) * input


class CVSigLog(nn.Module):
    r"""
    CVSigLog Activation Function.

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \frac{\mathbf{z}}{(c + 1/r * |\mathbf{z}|)}

    Based on work from the following paper:

        **G.M. Georgiou and C. Koutsougeras. Complex domain backpropagation.**

            - Eq. (20)

            - https://ieeexplore.ieee.org/abstract/document/142037
    """

    def __init__(self, c: float = 1.0, r: float = 1.0) -> None:
        super(CVSigLog, self).__init__()

        self.c = c
        self.r = r

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Computes the complex-valued SigLog activation function.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\frac{\mathbf{z}}{(c + 1/r * |\mathbf{z}|)}`
        """
        return input / (self.c + input.abs() / self.r)
