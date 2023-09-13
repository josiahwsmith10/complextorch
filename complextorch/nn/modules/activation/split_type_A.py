import torch.nn as nn

from .... import CVTensor
from ... import functional as cvF

__all__ = [
    "GeneralizedSplitActivation",
    "CVSplitTanh",
    "CTanh",
    "CVSplitSigmoid",
    "CSigmoid",
    "CVSplitAbs",
]


class GeneralizedSplitActivation(nn.Module):
    r"""
    Generalized Split *Type-A* Activation Function
    ----------------------------------------------

    Operates on the real and imaginary parts separately.

    Implements the operation:

    .. math::

        G(\mathbf{z}) = G_{real}(\mathbf{z}_{real}) + j G_{imag}(\mathbf{z}_{imag})

    *Type-A* nomenclature is defined in the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Section 4

            - https://arxiv.org/abs/2302.08286
    """

    def __init__(self, activation_r: nn.Module, activation_i: nn.Module) -> None:
        super(GeneralizedSplitActivation, self).__init__()
        self.activation_r = activation_r
        self.activation_i = activation_i

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Computes the generalized *Type-A* split activation function.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`\texttt{activation_r}(input.real) + j\texttt{activation_i}(input.imag)`
        """
        return cvF.apply_complex_split(self.activation_r, self.activation_i, input)


class CVSplitTanh(GeneralizedSplitActivation):
    r"""
    Split Complex-Valued Hyperbolic Tangent
    ---------------------------------------

    Implements the operation:

    .. math::

       G(\mathbf{z}) = \tanh(\mathbf{z}_{real}) + j \tanh(\mathbf{z}_{imag})

    Based on work from the following paper:

        **A Hirose, S Yoshida. Generalization characteristics of complex-valued feedforward neural networks in relation to signal coherence.**

            - Eq. (15)

            - https://ieeexplore.ieee.org/abstract/document/6138313
    """

    def __init__(self) -> None:
        super(CVSplitTanh, self).__init__(nn.Tanh(), nn.Tanh())


class CTanh(CVSplitTanh):
    r"""
    Alias for the :class:`CVSplitTanh`

    Implements the operation:

    .. math::

       G(\mathbf{z}) = \tanh(\mathbf{z}_{real}) + j \tanh(\mathbf{z}_{imag})

    Based on work from the following paper:

        **A Hirose, S Yoshida. Generalization characteristics of complex-valued feedforward neural networks in relation to signal coherence.**

            - Eq. (15)

            - https://ieeexplore.ieee.org/abstract/document/6138313
    """

    pass


class CVSplitSigmoid(GeneralizedSplitActivation):
    r"""
    Split Complex-Valued Sigmoid
    ----------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \text{sigmoid}(\mathbf{z}_{real}) + j \text{sigmoid}(\mathbf{z}_{imag})
    """

    def __init__(self) -> None:
        super(CVSplitSigmoid, self).__init__(nn.Sigmoid(), nn.Sigmoid())


class CSigmoid(CVSplitSigmoid):
    r"""
    Alias for the :class:`CVSplitSigmoid`

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \text{sigmoid}(\mathbf{z}_{real}) + j \text{sigmoid}(\mathbf{z}_{imag})
    """

    pass


class CVSplitAbs(nn.Module):
    r"""
    Split Absolute Value Activation Function.

    Implements the operation:

    .. math::

        G(\mathbf{z}) = |\mathbf{z}_{real}| + j |\mathbf{z}_{imag}|

    Based on work from the following paper:

        **A Marseet, F Sahin. Application of complex-valued convolutional neural network for next generation wireless networks.**

            - Section III-C

            - https://ieeexplore.ieee.org/abstract/document/8356260
    """

    def __init__(self) -> None:
        super(CVSplitAbs, self).__init__()

    def forward(self, input: CVTensor) -> CVTensor:
        r"""Computes the Type-A split abs() activation function.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`|\mathbf{z}_{real}| + j |\mathbf{z}_{imag}|`
        """
        return CVTensor(input.real.abs(), input.imag.abs())
