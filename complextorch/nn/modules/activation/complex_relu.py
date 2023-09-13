import torch.nn as nn

from .split_type_A import GeneralizedSplitActivation

__all__ = ["CVSplitReLU", "CReLU", "CPReLU"]


class CVSplitReLU(GeneralizedSplitActivation):
    r"""
    Split Complex-Valued Rectified Linear Unit
    ------------------------------------------

    Implements the operation:

    .. math::

        G(\mathbf{z}) = \texttt{ReLU}(\mathbf{z}_{real}) + j \texttt{ReLU}(\mathbf{z}_{imag})

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

        G(\mathbf{z}) = \texttt{ReLU}(\mathbf{z}_{real}) + j \texttt{ReLU}(\mathbf{z}_{imag})

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

        G(\mathbf{z}) = \texttt{PReLU}(\mathbf{z}_{real}) + j \texttt{PReLU}(\mathbf{z}_{imag})

    Based on work from the following paper:

        **H. Jing, S. Li, K. Miao, S. Wang, X. Cui, G. Zhao and H. Sun. Enhanced Millimeter-Wave 3-D Imaging via Complex-Valued Fully Convolutional Neural Network.**

            - Eq. (2)

            - https://www.mdpi.com/2079-9292/11/1/147
    """

    def __init__(self) -> None:
        super(CPReLU, self).__init__(nn.PReLU(), nn.PReLU())
