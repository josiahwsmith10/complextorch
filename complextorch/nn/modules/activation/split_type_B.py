import torch
import torch.nn as nn

from ... import functional as cvF

__all__ = [
    "GeneralizedPolarActivation",
    "CVPolarTanh",
    "CVPolarSquash",
    "modReLU",
    "CVPolarLog",
]


class GeneralizedPolarActivation(nn.Module):
    r"""
    Generalized Split *Type-B* Polar Activation Function
    ----------------------------------------------------

    Operates on the magnitude and phase separately. Often :math:`G_\angle(\angle\mathbf{z})` is the identity, in which case activation_phase should be set to None.

    Implements the operation:

    .. math::

        G(\mathbf{z}) = G_{||}(|\mathbf{z}|) \odot \exp(j G_\angle(\angle\mathbf{z}))

    `Type-B` activation function is defined in the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Section 4

            - https://arxiv.org/abs/2302.08286
    """

    def __init__(self, activation_mag: nn.Module, activation_phase: nn.Module) -> None:
        super(GeneralizedPolarActivation, self).__init__()
        self.activation_mag = activation_mag
        self.activation_phase = activation_phase

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes the generalized *Type-B* split activation function.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`G_{||}(|\mathbf{z}|) \odot \exp(j G_\angle(\angle\mathbf{z}))`
        """
        return cvF.apply_complex_polar(
            self.activation_mag, self.activation_phase, input
        )


class CVPolarTanh(GeneralizedPolarActivation):
    r"""
    Complex-Valued Polar Tanh Activation Function
    ---------------------------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \tanh(|z|) \odot \exp(j \angle\mathbf{z})

    *Note*: phase information is unchanged

    Based on work from the following paper:

        **A Hirose, S Yoshida. Generalization characteristics of complex-valued feedforward neural networks in relation to signal coherence**

            - Eq. (8)

            - https://ieeexplore.ieee.org/abstract/document/6138313
    """

    def __init__(self) -> None:
        super(CVPolarTanh, self).__init__(nn.Tanh(), None)


class _Squash(nn.Module):
    r"""
    Helper class to compute `squash` functionality on real-valued magnitude torch.Tensor.

    Implements the operation:

    .. math::

        G(x) = x^2 / (1 + x^2)
    """

    def __init__(self) -> None:
        super(_Squash, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes the squash functionality.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`x^2 / (1 + x^2)`
        """
        return input**2 / (1 + input**2)


class CVPolarSquash(GeneralizedPolarActivation):
    r"""
    Complex-Valued Polar Squash Activation Function
    -----------------------------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \frac{|\mathbf{z}|^2}{(1 + |\mathbf{z}|^2)} \odot \exp(j \angle\mathbf{z})

    Based on work from the following paper:

        **D Hayakawa, T Masuko, H Fujimura. Applying complex-valued neural networks to acoustic modeling for speech recognition.**

            - Section III-C

            - http://www.apsipa.org/proceedings/2018/pdfs/0001725.pdf
    """

    def __init__(self):
        super(CVPolarSquash, self).__init__(_Squash(), None)


class _LogXPlus1(nn.Module):
    r"""
    Helper class to compute :math:`\log(x + 1)` on real-valued magnitude torch.Tensor.

    Implements the operation:

    .. math::

        G(x) = \ln(x + 1)
    """

    def __init__(self) -> None:
        super(_LogXPlus1, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes the :math:`\log(x + 1)` functionality.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\ln(x + 1)`
        """
        return torch.log(input + 1)


class CVPolarLog(GeneralizedPolarActivation):
    r"""
    Complex-Valued Polar Squash Activation Function
    -----------------------------------------------

    Implements the operation

    .. math::

        G(\mathbf{z}) = \ln(|\mathbf{z}| + 1) \odot \exp(j \angle\mathbf{z})

    Based on work from the following paper:

        **D Hayakawa, T Masuko, H Fujimura. Applying complex-valued neural networks to acoustic modeling for speech recognition.**

            - Section III-C

            - http://www.apsipa.org/proceedings/2018/pdfs/0001725.pdf
    """

    def __init__(self) -> None:
        super(CVPolarLog, self).__init__(_LogXPlus1(), None)


class _modReLU(nn.Module):
    r"""
    Helper class to compute :math:`\text{ReLU}(x + b)` on real-valued magnitude torch.Tensor.

    Implements the operation:

    .. math::

        G(x) = \text{ReLU}(x + b)
    """

    def __init__(self, bias: float = 0.0) -> None:
        super(_modReLU, self).__init__()

        assert bias < 0, "bias must be smaller than 0 to have a non-linearity effect"

        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes the :math:`\text{ReLU}(x + b)` functionality.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: :math:`\text{ReLU}(x + b)`
        """
        return torch.relu(input + self.bias)


class modReLU(GeneralizedPolarActivation):
    r"""
    modulus Rectified Linear Unit
    -----------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \texttt{ReLU}(|\mathbf{z}| + b) \odot \frac{\mathbf{z}}{|\mathbf{z}|} = \texttt{ReLU}(|\mathbf{z}| + b) \odot \exp(j \angle\mathbf{z}).

    Notice that :math:`|\mathbf{z}|` (:math:`\mathbf{z}`.abs()) is always positive, so if :math:`b > 0`  then :math:`|\mathbf{z}| + b > = 0` always.
    In order to have any non-linearity effect, :math:`b` must be smaller than :math:`0` (:math:`b < 0`).

    Based on work from the following papers:

        **Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks.**

            - Eq. (8)

            - https://arxiv.org/abs/1511.06464



        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Eq. (36)

            - https://arxiv.org/abs/2302.08286
    """

    def __init__(self, bias: float = 0.0) -> None:
        assert bias < 0, "bias must be smaller than 0 to have a non-linearity effect"

        super(modReLU, self).__init__(_modReLU(bias), None)
