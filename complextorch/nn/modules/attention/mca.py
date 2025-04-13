import torch
import torch.nn as nn

from .... import nn as cvnn

__all__ = [
    "MaskedChannelAttention1d",
    "MaskedChannelAttention2d",
    "MaskedChannelAttention3d",
]


class _MaskedChannelAttention(nn.Module):
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
        super(_MaskedChannelAttention, self).__init__()
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get attention values
        attn = self.conv_up(self.act(self.conv_down(self.avg_pool(input))))

        # Compute mask
        mask = self.MaskingClass(attn)

        return input * mask


class MaskedChannelAttention1d(_MaskedChannelAttention):
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
        super(MaskedChannelAttention1d, self).__init__(
            channels=channels,
            reduction_factor=reduction_factor,
            MaskingClass=MaskingClass,
            act=act,
        )

        self.avg_pool = cvnn.AdaptiveAvgPool1d(1)

        self.conv_down = cvnn.SlowConv1d(
            in_channels=channels,
            out_channels=self.reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.conv_up = cvnn.SlowConv1d(
            in_channels=self.reduced_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )


class MaskedChannelAttention2d(_MaskedChannelAttention):
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
        super(MaskedChannelAttention2d, self).__init__(
            channels=channels,
            reduction_factor=reduction_factor,
            MaskingClass=MaskingClass,
            act=act,
        )

        self.avg_pool = cvnn.AdaptiveAvgPool2d(1)

        self.conv_down = cvnn.SlowConv2d(
            in_channels=channels,
            out_channels=self.reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.conv_up = cvnn.SlowConv2d(
            in_channels=self.reduced_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )


class MaskedChannelAttention3d(_MaskedChannelAttention):
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
        super(MaskedChannelAttention3d, self).__init__(
            channels=channels,
            reduction_factor=reduction_factor,
            MaskingClass=MaskingClass,
            act=act,
        )

        self.avg_pool = cvnn.AdaptiveAvgPool3d(1)

        self.conv_down = cvnn.SlowConv3d(
            in_channels=channels,
            out_channels=self.reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.conv_up = cvnn.SlowConv3d(
            in_channels=self.reduced_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
