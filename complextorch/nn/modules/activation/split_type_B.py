import torch
import torch.nn as nn

from .... import CVTensor
from ... import functional as cvF

__all__ = ["GeneralizedPolarActivation", "CVPolarTanh", "CVPolarSquash", "CVPolarLog"]


class GeneralizedPolarActivation(nn.Module):
    """
    Generalized Split *Type-B* Polar Activation Function
    ----------------------------------------------------

    Operates on the magnitude and phase separately.

    Implements the operation:

    .. math::

        G(\mathbf{z}) = G_{mag}(|\mathbf{z}|) * \exp(j G_{phase}(\\text{angle}(\mathbf{z})))

    `Type-B` activation function is defined in the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Section 4

            - https://arxiv.org/abs/2302.08286
    """

    def __init__(self, activation_mag: nn.Module, activation_phase: nn.Module) -> None:
        super(GeneralizedPolarActivation, self).__init__()
        self.activation_mag = activation_mag
        self.activation_phase = (
            activation_phase if activation_phase is not None else nn.Identity()
        )

    def forward(self, input: CVTensor) -> CVTensor:
        """Computes the generalized *Type-B* split activation function.

        Args:
            input (CVTensor): input tensor

        Returns:
            CVTensor: :math:`G_{mag}(|\mathbf{z}|) * \exp(j G_{phase}(\\text{angle}(\mathbf{z})))`
        """
        return cvF.apply_complex_polar(
            self.activation_mag, self.activation_phase, input
        )


class CVPolarTanh(GeneralizedPolarActivation):
    """
    Complex-Valued Polar Tanh Activation Function
    ---------------------------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \\tanh(|z|) * \exp(j \\text{angle}(\mathbf{z}))

    *Note*: phase information is unchanged

    Based on work from the following paper:

        **A Hirose, S Yoshida. Generalization characteristics of complex-valued feedforward neural networks in relation to signal coherence**

            - Eq. (15)

            - https://ieeexplore.ieee.org/abstract/document/6138313
    """

    def __init__(self):
        super(CVPolarTanh, self).__init__(nn.Tanh(), None)


class _Squash(nn.Module):
    """
    Helper class to compute `squash` functionality on real-valued magnitude torch.Tensor.

    Implements the operation:

    .. math::

        G(x) = x^2 / (1 + x^2)
    """

    def __init__(self) -> None:
        super(_Squash, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Computes the squash functionality.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`x^2 / (1 + x^2)`
        """
        return input**2 / (1 + input**2)


class CVPolarSquash(GeneralizedPolarActivation):
    """
    Complex-Valued Polar Squash Activation Function
    -----------------------------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = |\mathbf{z}|^2 / (1 + |\mathbf{z}|^2) * \exp(j \\text{angle}(\mathbf{z}))

    Based on work from the following paper:

        **D Hayakawa, T Masuko, H Fujimura. Applying complex-valued neural networks to acoustic modeling for speech recognition.**

            - Section III-C

            - http://www.apsipa.org/proceedings/2018/pdfs/0001725.pdf
    """

    def __init__(self):
        super(CVPolarSquash, self).__init__(_Squash(), None)


class _LogXPlus1(nn.Module):
    """
    Helper class to compute :math:`\log(x + 1)` on real-valued magnitude torch.Tensor.

    Implements the operation:

    .. math::

        G(x) = \ln(x + 1)
    """

    def __init__(self) -> None:
        super(_LogXPlus1, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Computes the :math:`\log(x + 1)` functionality.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\ln(x + 1)`
        """
        return torch.log(input + 1)


class CVPolarLog(GeneralizedPolarActivation):
    """
    Complex-Valued Polar Squash Activation Function
    -----------------------------------------------

    Implements the operation

    .. math::

        G(\mathbf{z}) = \ln(|\mathbf{z}| + 1) * \exp(j \\text{angle}(\mathbf{z}))

    Based on work from the following paper:

        **D Hayakawa, T Masuko, H Fujimura. Applying complex-valued neural networks to acoustic modeling for speech recognition.**

            - Section III-C

            - http://www.apsipa.org/proceedings/2018/pdfs/0001725.pdf
    """

    def __init__(self):
        super(CVPolarLog, self).__init__(_LogXPlus1(), None)
