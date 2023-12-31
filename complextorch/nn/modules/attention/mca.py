import torch.nn as nn

from .... import CVTensor
from .... import nn as cvnn

__all__ = [
    "CVMaskedChannelAttention1d",
    "CVMaskedChannelAttention2d",
    "CVMaskedChannelAttention3d",
]


class _CVMaskedChannelAttention(nn.Module):
    r"""
    Complex-Valued Masked Channel Attention (MCA) Base Module
    ---------------------------------------------------------
    
    Implements the operation:

    .. math::

        \texttt{CV-MCA}(\mathbf{z}) = \mathcal{M}(H_\text{ConvUp}(\mathcal{A}(H_\text{ConvDown}(\mathbf{z})))) \odot \mathbf{z},

    where :math:`\mathcal{M}(\cdot)` is the masking function (by default, ComplexRatioMask is used) and :math:`H_\text{ConvUp}(\cdot)` and :math:`H_\text{ConvDown}(\cdot)` are N-D convolution layers with kernel sizes of 1 that reduce the channel dimension by a factor :math:`r`.

    """

    def __init__(
        self,
        channels: int,
        reduction_factor: int = 2,
        MaskingClass: nn.Module = cvnn.ComplexRatioMask,
        act: nn.Module = cvnn.CReLU,
    ) -> None:
        super(_CVMaskedChannelAttention, self).__init__()
        self.channels = channels
        self.reduction_factor = reduction_factor
        self.MaskingClass = MaskingClass()
        self.act = act()

        assert (
            channels % reduction_factor == 0
        ), "Channels / Reduction Factor must yield integer"
        self.reduced_channels = int(channels / reduction_factor)

        # Placeholders
        self.avg_pool = None
        self.conv_down = None
        self.conv_up = None

    def forward(self, input: CVTensor) -> CVTensor:
        # Get attention values
        attn = self.conv_up(self.act(self.conv_down(self.avg_pool(input))))

        # Compute mask
        mask = self.MaskingClass(attn)

        return input * mask


class CVMaskedChannelAttention1d(_CVMaskedChannelAttention):
    r"""
    1-D Complex-Valued Masked Channel Attention (MCA) Module
    --------------------------------------------------------

    Generalized for arbitrary masking function (see :doc:`mask <../mask>` for implemented masking functions)
    
    Implements the operation:

    .. math::

        \texttt{CV-MCA}(\mathbf{z}) = \mathcal{M}(H_\text{ConvUp}(\mathcal{A}(H_\text{ConvDown}(\mathbf{z})))) \odot \mathbf{z},

    where :math:`\mathcal{M}(\cdot)` is the masking function (by default, ComplexRatioMask is used) and :math:`H_\text{ConvUp}(\cdot)` and :math:`H_\text{ConvDown}(\cdot)` are 1-D convolution layers with kernel sizes of 1 that reduce the channel dimension by a factor :math:`r`.

    Based on work from the following paper:

        **HW Cho, S Choi, YR Cho, and J Kim: Complex-Valued Channel Attention and Application in Ego-Velocity Estimation With Automotive Radar**

            - Fig. 3

            - https://ieeexplore.ieee.org/abstract/document/9335579
    """

    def __init__(
        self,
        channels: int,
        reduction_factor: int = 2,
        MaskingClass: nn.Module = cvnn.ComplexRatioMask,
        act: nn.Module = cvnn.CReLU,
    ) -> None:
        super(CVMaskedChannelAttention1d, self).__init__(
            channels=channels,
            reduction_factor=reduction_factor,
            MaskingClass=MaskingClass,
            act=act,
        )

        self.avg_pool = cvnn.CVAdaptiveAvgPool1d(1)

        self.conv_down = cvnn.CVConv1d(
            in_channels=channels,
            out_channels=self.reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.conv_up = cvnn.CVConv1d(
            in_channels=self.reduced_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )


class CVMaskedChannelAttention2d(_CVMaskedChannelAttention):
    r"""
    2-D Complex-Valued Masked Channel Attention (MCA) Module
    --------------------------------------------------------
    
    Implements the operation:

    .. math::

        \texttt{CV-MCA}(\mathbf{z}) = \mathcal{M}(H_\text{ConvUp}(\mathcal{A}(H_\text{ConvDown}(\mathbf{z})))) \odot \mathbf{z},

    where :math:`\mathcal{M}(\cdot)` is the masking function (by default, ComplexRatioMask is used) and :math:`H_\text{ConvUp}(\cdot)` and :math:`H_\text{ConvDown}(\cdot)` are 2-D convolution layers with kernel sizes of 1 that reduce the channel dimension by a factor :math:`r`.

    Generalized for arbitrary masking function (see :doc:`mask <../mask>` for implemented masking functions)

    Based on work from the following paper:

        **HW Cho, S Choi, YR Cho, and J Kim: Complex-Valued Channel Attention and Application in Ego-Velocity Estimation With Automotive Radar**

            - Fig. 3

            - https://ieeexplore.ieee.org/abstract/document/9335579
    """

    def __init__(
        self,
        channels: int,
        reduction_factor: int = 2,
        MaskingClass: nn.Module = cvnn.ComplexRatioMask,
        act: nn.Module = cvnn.CReLU,
    ) -> None:
        super(CVMaskedChannelAttention2d, self).__init__(
            channels=channels,
            reduction_factor=reduction_factor,
            MaskingClass=MaskingClass,
            act=act,
        )

        self.avg_pool = cvnn.CVAdaptiveAvgPool2d(1)

        self.conv_down = cvnn.CVConv2d(
            in_channels=channels,
            out_channels=self.reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.conv_up = cvnn.CVConv2d(
            in_channels=self.reduced_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )


class CVMaskedChannelAttention3d(_CVMaskedChannelAttention):
    r"""
    3-D Complex-Valued Masked Channel Attention (MCA) Module
    --------------------------------------------------------

    Generalized for arbitrary masking function (see :doc:`mask <../mask>` for implemented masking functions)
    
    Implements the operation:

    .. math::

        \texttt{CV-MCA}(\mathbf{z}) = \mathcal{M}(H_\text{ConvUp}(\mathcal{A}(H_\text{ConvDown}(\mathbf{z})))) \odot \mathbf{z},

    where :math:`\mathcal{M}(\cdot)` is the masking function (by default, ComplexRatioMask is used) and :math:`H_\text{ConvUp}(\cdot)` and :math:`H_\text{ConvDown}(\cdot)` are 3-D convolution layers with kernel sizes of 1 that reduce the channel dimension by a factor :math:`r`.


    Based on work from the following paper:

        **HW Cho, S Choi, YR Cho, and J Kim: Complex-Valued Channel Attention and Application in Ego-Velocity Estimation With Automotive Radar**

            - Fig. 3

            - https://ieeexplore.ieee.org/abstract/document/9335579
    """

    def __init__(
        self,
        channels: int,
        reduction_factor: int = 2,
        MaskingClass: nn.Module = cvnn.ComplexRatioMask,
        act: nn.Module = cvnn.CReLU,
    ) -> None:
        super(CVMaskedChannelAttention3d, self).__init__(
            channels=channels,
            reduction_factor=reduction_factor,
            MaskingClass=MaskingClass,
            act=act,
        )

        self.avg_pool = cvnn.CVAdaptiveAvgPool3d(1)

        self.conv_down = cvnn.CVConv3d(
            in_channels=channels,
            out_channels=self.reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.conv_up = cvnn.CVConv3d(
            in_channels=self.reduced_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
