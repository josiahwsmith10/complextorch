import torch.nn as nn

from .split_type_A import GeneralizedSplitActivation

__all__ = ["CVSplitReLU", "CReLU", "CPReLU"]


class CVSplitReLU(GeneralizedSplitActivation):
    """
    Split Complex-Valued Rectified Linear Unit.

    Jingkun Gao, Bin Deng, Yuliang Qin, Hongqiang Wang and Xiang Li. Enhanced Radar Imaging Using a Complex-valued Convolutional Neural Network.
    Eq. (5)
    https://arxiv.org/abs/1712.10096
    """

    def __init__(self, inplace: bool = True) -> None:
        super(CVSplitReLU, self).__init__(nn.ReLU(inplace), nn.ReLU(inplace))


class CReLU(CVSplitReLU):
    pass


class CPReLU(GeneralizedSplitActivation):
    """
    Split Complex-Valed Parametric Rectified Linear Unit.

    H. Jing, S. Li, K. Miao, S. Wang, X. Cui, G. Zhao and H. Sun. Enhanced Millimeter-Wave 3-D Imaging via Complex-Valued Fully Convolutional Neural Network.
    Eq. (2)
    https://www.mdpi.com/2079-9292/11/1/147
    """

    def __init__(self) -> None:
        super(CPReLU, self).__init__(nn.PReLU(), nn.PReLU())
