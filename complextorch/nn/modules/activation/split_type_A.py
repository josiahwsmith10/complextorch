import torch.nn as nn

from .... import CVTensor
from ... import functional as cvF

__all__ = ['GeneralizedSplitActivation', 'CVSplitTanh', 'CTanh', 'CVSplitSigmoid', 'CSigmoid', 'CVSplitAbs']

class GeneralizedSplitActivation(nn.Module):
    """
    Generalized Split Activation Function.

    Operates on the real and imaginary parts separately.

    f(x) = f_r(x_r) + 1j * f_i(x_i)

    `Type-A` activation function is defined in the following paper:
    J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.
    Section 4
    https://arxiv.org/abs/2302.08286
    """

    def __init__(self, activation_r: nn.Module, activation_i: nn.Module) -> None:
        super(GeneralizedSplitActivation, self).__init__()
        self.activation_r = activation_r
        self.activation_i = activation_i

    def forward(self, input: CVTensor) -> CVTensor:
        return cvF.apply_complex_split(self.activation_r, self.activation_i, input)


class CVSplitTanh(GeneralizedSplitActivation):
    """
    Split Complex-Valued Hyperbolic Tangent.
    
    CVSplitTanh(z) = tanh(z_r) + 1j * tanh(z_i)
    
    A Hirose, S Yoshida. Generalization characteristics of complex-valued feedforward neural networks in relation to signal coherence
    Eq. (15)
    https://ieeexplore.ieee.org/abstract/document/6138313
    """

    def __init__(self) -> None:
        super(CVSplitTanh, self).__init__(nn.Tanh(), nn.Tanh())


class CTanh(CVSplitTanh):
    pass


class CVSplitSigmoid(GeneralizedSplitActivation):
    """
    Split Complex-Valued Sigmoid.
    
    CVSplitSigmoid(z) = sigmoid(z_r) + 1j * sigmoid(z_i)
    """

    def __init__(self) -> None:
        super(CVSplitSigmoid, self).__init__(nn.Sigmoid(), nn.Sigmoid())


class CSigmoid(CVSplitSigmoid):
    pass


class CVSplitAbs(nn.Module):
    """
    Split Absolute Value Activation Function.
    
    CVSplitAbs(z) = abs(z_r) + 1j * abs(z_i)
    
    A Marseet, F Sahin. Application of complex-valued convolutional neural network for next generation wireless networks.
    Section III-C
    https://ieeexplore.ieee.org/abstract/document/8356260
    """
    
    def __init__(self) -> None:
        super(CVSplitAbs, self).__init__()
        
    def forward(self, input: CVTensor) -> CVTensor:
        return CVTensor(input.real.abs(), input.imag.abs())
        