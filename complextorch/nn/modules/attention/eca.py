import numpy as np
import torch.nn as nn

from .... import CVTensor
from .... import nn as cvnn

__all__ = [
    "CVEfficientChannelAttention1d",
    "CVEfficientChannelAttention2d",
    "CVEfficientChannelAttention3d",
]


class _CVEfficientChannelAttention(nn.Module):
    r"""
    Complex-Valued Efficient Channel Attention Base Class
    -----------------------------------------------------

    Implements the operation:

    .. math::

        \texttt{CV-ECA}(\mathbf{z}) = \mathcal{M}(\text{conv}(H_\texttt{CVAdaptiveAvgPoolNd}(\mathbf{z}))) \odot \mathbf{z},

    where :math:`\mathcal{M}(\cdot)` is the masking function (by default, ComplexRatioMask is used) and :math:`H_\texttt{CVAdaptiveAvgPoolNd}(\cdot)` is the complex-valued global :doc:`pooling <../pooling>` operator.
    """

    def __init__(
        self,
        channels: int,
        MaskingClass: nn.Module,
        AvgPoolClass: nn.Module,
        b: int = 1,
        gamma: int = 2,
    ) -> None:
        super(_CVEfficientChannelAttention, self).__init__()
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.avg_pool = AvgPoolClass(1)
        self.mask = MaskingClass()
        self.conv = cvnn.CVConv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size(),
            padding=(self.kernel_size() - 1) // 2,
            bias=False,
        )

    def kernel_size(self) -> int:
        k = int(abs((np.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, input: CVTensor) -> CVTensor:
        batch_size, channels, *im_size = input.shape
        one_vec = [1] * len(im_size)

        # feature descriptor on the global spatial information
        y = self.avg_pool(input)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).view(batch_size, 1, channels)).transpose(-1, -2)
        y = y.transpose(-1, -2).view(batch_size, channels, *one_vec)

        # Multi-scale information fusion
        y = self.mask(y)

        return input * y


class CVEfficientChannelAttention1d(_CVEfficientChannelAttention):
    r"""
    1-D Complex-Valued Efficient Channel Attention
    ----------------------------------------------

    Implements the operation:

    .. math::

        \texttt{CV-ECA}(\mathbf{z}) = \mathcal{M}(\text{conv}(H_\texttt{CVAdaptiveAvgPool1d}(\mathbf{z}))) \odot \mathbf{z},

    where :math:`\mathcal{M}(\cdot)` is the masking function (by default, ComplexRatioMask is used) and :math:`H_\texttt{CVAdaptiveAvgPoolNd}(\cdot)` is the complex-valued global :doc:`pooling <../pooling>` operator.

    Based on work from the following paper:

        **Q Wang, B Wu, P Zhu, P Li, W Zuo, and Q Hu: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks**

            - Fig. 2

            - https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf
    """

    def __init__(
        self,
        channels: int,
        MaskingClass: nn.Module = cvnn.ComplexRatioMask,
        b: int = 1,
        gamma: int = 2,
    ) -> None:
        super(CVEfficientChannelAttention1d, self).__init__(
            channels=channels,
            MaskingClass=MaskingClass,
            AvgPoolClass=cvnn.CVAdaptiveAvgPool1d,
            b=b,
            gamma=gamma,
        )


class CVEfficientChannelAttention2d(_CVEfficientChannelAttention):
    r"""
    2-D Complex-Valued Efficient Channel Attention
    ----------------------------------------------

    Implements the operation:

    .. math::

        \texttt{CV-ECA}(\mathbf{z}) = \mathcal{M}(\text{conv}(H_\texttt{CVAdaptiveAvgPool2d}(\mathbf{z}))) \odot \mathbf{z},

    where :math:`\mathcal{M}(\cdot)` is the masking function (by default, ComplexRatioMask is used) and :math:`H_\texttt{CVAdaptiveAvgPoolNd}(\cdot)` is the complex-valued global :doc:`pooling <../pooling>` operator.

    Based on work from the following paper:

        **Q Wang, B Wu, P Zhu, P Li, W Zuo, and Q Hu: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks**

            - Fig. 2

            - https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf
    """

    def __init__(
        self,
        channels: int,
        MaskingClass: nn.Module = cvnn.ComplexRatioMask,
        b: int = 1,
        gamma: int = 2,
    ) -> None:
        super(CVEfficientChannelAttention2d, self).__init__(
            channels=channels,
            MaskingClass=MaskingClass,
            AvgPoolClass=cvnn.CVAdaptiveAvgPool2d,
            b=b,
            gamma=gamma,
        )


class CVEfficientChannelAttention3d(_CVEfficientChannelAttention):
    r"""
    3-D Complex-Valued Efficient Channel Attention
    ----------------------------------------------

    Implements the operation:

    .. math::

        \texttt{CV-ECA}(\mathbf{z}) = \mathcal{M}(\text{conv}(H_\texttt{CVAdaptiveAvgPool3d}(\mathbf{z}))) \odot \mathbf{z},

    where :math:`\mathcal{M}(\cdot)` is the masking function (by default, ComplexRatioMask is used) and :math:`H_\texttt{CVAdaptiveAvgPoolNd}(\cdot)` is the complex-valued global :doc:`pooling <../pooling>` operator.

    Based on work from the following paper:

        **Q Wang, B Wu, P Zhu, P Li, W Zuo, and Q Hu: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks**

            - Fig. 2

            - https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf
    """

    def __init__(
        self,
        channels: int,
        MaskingClass: nn.Module = cvnn.ComplexRatioMask,
        b: int = 1,
        gamma: int = 2,
    ) -> None:
        super(CVEfficientChannelAttention3d, self).__init__(
            channels=channels,
            MaskingClass=MaskingClass,
            AvgPoolClass=cvnn.CVAdaptiveAvgPool3d,
            b=b,
            gamma=gamma,
        )
