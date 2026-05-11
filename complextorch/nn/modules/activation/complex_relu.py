import torch
import torch.nn as nn

from complextorch.nn.modules.activation.split_type_A import GeneralizedSplitActivation

__all__ = ["CVSplitReLU", "CReLU", "CPReLU", "zAbsReLU", "zLeakyReLU"]


class CVSplitReLU(GeneralizedSplitActivation):
    r"""
    Split Complex-Valued Rectified Linear Unit
    ------------------------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \texttt{ReLU}(\mathbf{x}) + j \texttt{ReLU}(\mathbf{y})

    Based on work from the following paper:

        **Jingkun Gao, Bin Deng, Yuliang Qin, Hongqiang Wang and Xiang Li. Enhanced Radar Imaging Using a Complex-valued Convolutional Neural Network.**

            - Eq. (5)

            - https://arxiv.org/abs/1712.10096
    """

    def __init__(self, inplace: bool = True) -> None:
        super(CVSplitReLU, self).__init__(nn.ReLU(inplace), nn.ReLU(inplace))


class CReLU(CVSplitReLU):
    r"""
    Split Complex-Valued Rectified Linear Unit
    ------------------------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \texttt{ReLU}(\mathbf{x}) + j \texttt{ReLU}(\mathbf{y})

    Alias for :class:`CVSplitReLU`. The nomenclature CReLU is used only in certain literature to denote the split complex-valued rectified linear unit.
    """

    pass


class CPReLU(GeneralizedSplitActivation):
    r"""
    Split Complex-Valued Parametric Rectified Linear Unit
    -----------------------------------------------------

    Split Type-A extension of the `Parametric ReLU <https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html>`_ for complex-valued tensors.

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \texttt{PReLU}(\mathbf{x}) + j \texttt{PReLU}(\mathbf{y})

    Based on work from the following paper:

        **H. Jing, S. Li, K. Miao, S. Wang, X. Cui, G. Zhao and H. Sun. Enhanced Millimeter-Wave 3-D Imaging via Complex-Valued Fully Convolutional Neural Network.**

            - Eq. (2)

            - https://www.mdpi.com/2079-9292/11/1/147
    """

    def __init__(self) -> None:
        super(CPReLU, self).__init__(nn.PReLU(), nn.PReLU())


class zAbsReLU(nn.Module):
    r"""
    Magnitude-Thresholded ReLU with Learnable Threshold
    ---------------------------------------------------

    Zeros out elements whose magnitude is below a learnable threshold
    :math:`a`, preserving the phase of passing elements:

    .. math::

        \texttt{zAbsReLU}(z) = \begin{cases}
            z & \text{if } |z| \geq a \\
            0 & \text{otherwise}
        \end{cases}

    Args:
        a_init: initial value of the (scalar) threshold parameter.
    """

    def __init__(self, a_init: float = 0.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mask = (input.abs() >= self.a).to(
            input.dtype if input.is_complex() else input.real.dtype
        )
        if input.is_complex():
            return input * mask
        return input * mask


class zLeakyReLU(nn.Module):
    r"""
    Leaky First-Quadrant Complex ReLU
    ---------------------------------

    Soft version of :class:`zReLU`: passes :math:`z` unchanged when both
    :math:`\Re z > 0` and :math:`\Im z > 0`; scales by ``negative_slope``
    elsewhere.

    .. math::

        \texttt{zLeakyReLU}(z) = \begin{cases}
            z & \text{if } \Re z > 0 \text{ and } \Im z > 0 \\
            \alpha\, z & \text{otherwise}
        \end{cases}
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_q1 = (input.real > 0) & (input.imag > 0)
        scale = torch.where(
            in_q1,
            torch.ones_like(input.real),
            torch.full_like(input.real, self.negative_slope),
        )
        return input * scale

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"
