import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]


class _Conv(nn.Module):
    r"""
    torch.Tensor-based Complex-Valued Convolution
    -----------------------------------------
    """

    def __init__(
        self,
        ConvClass: nn.Module,
        ConvFunc,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.ConvFunc = ConvFunc
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Assumes PyTorch complex weight initialization is correct
        __temp = ConvClass(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype if dtype else torch.cfloat,
        )

        self.conv_r = ConvClass(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.conv_i = ConvClass(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.conv_r.weight.data = __temp.weight.real
        self.conv_i.weight.data = __temp.weight.imag

        if bias:
            self.bias_r = nn.Parameter(__temp.bias.real.detach().clone())
            self.bias_i = nn.Parameter(__temp.bias.imag.detach().clone())
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

    @property
    def weight(self) -> torch.Tensor:
        return torch.complex(self.conv_r.weight, self.conv_i.weight)

    @property
    def bias(self) -> torch.Tensor:
        if self.bias_r is None:
            return None
        return torch.complex(self.bias_r, self.bias_i)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes convolution 25% faster than naive method by using Gauss' multiplication trick
        """
        t1 = self.conv_r(input.real)
        t2 = self.conv_i(input.imag)
        t3 = self.ConvFunc(
            input=(input.real + input.imag),
            weight=(self.conv_r.weight + self.conv_i.weight),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        out_r = t1 - t2
        out_i = t3 - t2 - t1
        if self.bias_r is not None:
            bias_shape = (-1,) + (1,) * (out_r.dim() - 2)
            out_r = out_r + self.bias_r.view(bias_shape)
            out_i = out_i + self.bias_i.view(bias_shape)
        return torch.complex(out_r, out_i)


class Conv1d(_Conv):
    r"""
    1-D Complex-Valued Convolution
    ------------------------------

    Based on the `PyTorch torch.nn.Conv1d <https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html>`_ implementation.

    Employs Gauss' multiplication trick to reduce number of computations by 25% compare with the naive implementation.

    The most common implementation of complex-valued convolution entails the following computation:

    .. math::

        G(\mathbf{z}) = \text{conv}(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})) - \text{conv}(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I})) + j(\text{conv}(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I})) + \text{conv}(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})))

    where :math:`\mathbf{W}` and :math:`\mathbf{b}` are the complex-valued weight and bias tensors, respectively, and :math:`\text{conv}(\cdot)` is the conovlution operator.

    By comparison, using Gauss' trick, the complex-vauled convolution can be implemented as:

    .. math::

        t1 =& \text{conv}(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R}))

        t2 =& \text{conv}(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I}))

        t3 =& \text{conv}(\mathbf{z}_\mathbb{R} + \mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R} + \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{R} + \mathbf{b}_\mathbb{I}))

        G(\mathbf{z}) =& t1 - t2 + j(t3 - t2 - t1)

    requiring only 3 convolution operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            ConvClass=nn.Conv1d,
            ConvFunc=F.conv1d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )


class Conv2d(_Conv):
    r"""
    2-D Complex-Valued Convolution
    ------------------------------

    Based on the `PyTorch torch.nn.Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_ implementation.

    Employs Gauss' multiplication trick to reduce number of computations by 25% compare with the naive implementation.

    The most common implementation of complex-valued convolution entails the following computation:

    .. math::

        G(\mathbf{z}) = \text{conv}(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})) - \text{conv}(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I})) + j(\text{conv}(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I})) + \text{conv}(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})))

    where :math:`\mathbf{W}` and :math:`\mathbf{b}` are the complex-valued weight and bias tensors, respectively, and :math:`\text{conv}(\cdot)` is the conovlution operator.

    By comparison, using Gauss' trick, the complex-vauled convolution can be implemented as:

    .. math::

        t1 =& \text{conv}(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R}))

        t2 =& \text{conv}(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I}))

        t3 =& \text{conv}(\mathbf{z}_\mathbb{R} + \mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R} + \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{R} + \mathbf{b}_\mathbb{I}))

        G(\mathbf{z}) =& t1 - t2 + j(t3 - t2 - t1)

    requiring only 3 convolution operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            ConvClass=nn.Conv2d,
            ConvFunc=F.conv2d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )


class Conv3d(_Conv):
    r"""
    3-D Complex-Valued Convolution
    ------------------------------

    Based on the `PyTorch torch.nn.Conv3d <https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html>`_ implementation.

    Employs Gauss' multiplication trick to reduce number of computations by 25% compare with the naive implementation.

    The most common implementation of complex-valued convolution entails the following computation:

    .. math::

        G(\mathbf{z}) = \text{conv}(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})) - \text{conv}(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I})) + j(\text{conv}(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I})) + \text{conv}(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})))

    where :math:`\mathbf{W}` and :math:`\mathbf{b}` are the complex-valued weight and bias tensors, respectively, and :math:`\text{conv}(\cdot)` is the conovlution operator.

    By comparison, using Gauss' trick, the complex-vauled convolution can be implemented as:

    .. math::

        t1 =& \text{conv}(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R}))

        t2 =& \text{conv}(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I}))

        t3 =& \text{conv}(\mathbf{z}_\mathbb{R} + \mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R} + \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{R} + \mathbf{b}_\mathbb{I}))

        G(\mathbf{z}) =& t1 - t2 + j(t3 - t2 - t1)

    requiring only 3 convolution operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: str | _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            ConvClass=nn.Conv3d,
            ConvFunc=F.conv3d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )


class _ConvTranspose(nn.Module):
    r"""
    torch.Tensor-based Complex-Valued Transposed Convolution
    ----------------------------------------------------
    """

    def __init__(
        self,
        ConvClass: nn.Module,
        ConvFunc,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.ConvFunc = ConvFunc
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups

        # Assumes PyTorch complex weight initialization is correct
        __temp = ConvClass(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype if dtype else torch.cfloat,
        )

        self.convt_r = ConvClass(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=False,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.convt_i = ConvClass(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=False,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.convt_r.weight.data = __temp.weight.real
        self.convt_i.weight.data = __temp.weight.imag

        if bias:
            self.bias_r = nn.Parameter(__temp.bias.real.detach().clone())
            self.bias_i = nn.Parameter(__temp.bias.imag.detach().clone())
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

    @property
    def weight(self) -> torch.Tensor:
        return torch.complex(self.convt_r.weight, self.convt_i.weight)

    @property
    def bias(self) -> torch.Tensor:
        if self.bias_r is None:
            return None
        return torch.complex(self.bias_r, self.bias_i)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes convolution 25% faster than naive method by using Gauss' multiplication trick
        """
        t1 = self.convt_r(input.real)
        t2 = self.convt_i(input.imag)
        t3 = self.ConvFunc(
            input=(input.real + input.imag),
            weight=(self.convt_r.weight + self.convt_i.weight),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        out_r = t1 - t2
        out_i = t3 - t2 - t1
        if self.bias_r is not None:
            bias_shape = (-1,) + (1,) * (out_r.dim() - 2)
            out_r = out_r + self.bias_r.view(bias_shape)
            out_i = out_i + self.bias_i.view(bias_shape)
        return torch.complex(out_r, out_i)


class ConvTranspose1d(_ConvTranspose):
    r"""
    1-D Complex-Valued Transposed Convolution
    -----------------------------------------

    Based on the `PyTorch torch.nn.ConvTranspose1d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html>`_ implementation.

    Employs Gauss' multiplication trick to reduce number of computations by 25% compare with the naive implementation.

    The most common implementation of complex-valued convolution entails the following computation:

    .. math::

        G(\mathbf{z}) = \text{conv}_T(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})) - \text{conv}_T(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I}))
        + j(\text{conv}_T(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I})) + \text{conv}_T(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})))

    where :math:`\mathbf{W}` and :math:`\mathbf{b}` are the complex-valued weight and bias tensors, respectively, and :math:`\text{conv}_T(\cdot)` is the transposed conovlution operator.

    By comparison, using Gauss' trick, the complex-vauled convolution can be implemented as:

    .. math::

        t1 =& \text{conv}_T(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R}))

        t2 =& \text{conv}_T(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I}))

        t3 =& \text{conv}_T(\mathbf{z}_\mathbb{R} + \mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R} + \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{R} + \mathbf{b}_\mathbb{I}))

        G(\mathbf{z}) =& t1 - t2 + j(t3 - t2 - t1)

    requiring only 3 transposed convolution operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            ConvClass=nn.ConvTranspose1d,
            ConvFunc=F.conv_transpose1d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )


class ConvTranspose2d(_ConvTranspose):
    r"""
    2-D Complex-Valued Transposed Convolution
    -----------------------------------------

    Based on the `PyTorch torch.nn.ConvTranspose2d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html>`_ implementation.

    Employs Gauss' multiplication trick to reduce number of computations by 25% compare with the naive implementation.

    The most common implementation of complex-valued convolution entails the following computation:

    .. math::

        G(\mathbf{z}) = \text{conv}_T(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})) - \text{conv}_T(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I}))
        + j(\text{conv}_T(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I})) + \text{conv}_T(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})))

    where :math:`\mathbf{W}` and :math:`\mathbf{b}` are the complex-valued weight and bias tensors, respectively, and :math:`\text{conv}_T(\cdot)` is the transposed conovlution operator.

    By comparison, using Gauss' trick, the complex-vauled convolution can be implemented as:

    .. math::

        t1 =& \text{conv}_T(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R}))

        t2 =& \text{conv}_T(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I}))

        t3 =& \text{conv}_T(\mathbf{z}_\mathbb{R} + \mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R} + \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{R} + \mathbf{b}_\mathbb{I}))

        G(\mathbf{z}) =& t1 - t2 + j(t3 - t2 - t1)

    requiring only 3 transposed convolution operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            ConvClass=nn.ConvTranspose2d,
            ConvFunc=F.conv_transpose2d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )


class ConvTranspose3d(_ConvTranspose):
    r"""
    3-D Complex-Valued Transposed Convolution
    -----------------------------------------

    Based on the `PyTorch torch.nn.ConvTranspose3d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html>`_ implementation.

    Employs Gauss' multiplication trick to reduce number of computations by 25% compare with the naive implementation.

    The most common implementation of complex-valued convolution entails the following computation:

    .. math::

        G(\mathbf{z}) = \text{conv}_T(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})) - \text{conv}_T(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I}))
        + j(\text{conv}_T(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I})) + \text{conv}_T(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R})))

    where :math:`\mathbf{W}` and :math:`\mathbf{b}` are the complex-valued weight and bias tensors, respectively, and :math:`\text{conv}_T(\cdot)` is the transposed conovlution operator.

    By comparison, using Gauss' trick, the complex-vauled convolution can be implemented as:

    .. math::

        t1 =& \text{conv}_T(\mathbf{z}_\mathbb{R}, \mathbf{W}_\mathbb{R}, \mathbf{b}_\mathbb{R}))

        t2 =& \text{conv}_T(\mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{I}))

        t3 =& \text{conv}_T(\mathbf{z}_\mathbb{R} + \mathbf{z}_\mathbb{I}, \mathbf{W}_\mathbb{R} + \mathbf{W}_\mathbb{I}, \mathbf{b}_\mathbb{R} + \mathbf{b}_\mathbb{I}))

        G(\mathbf{z}) =& t1 - t2 + j(t3 - t2 - t1)

    requiring only 3 transposed convolution operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            ConvClass=nn.ConvTranspose3d,
            ConvFunc=F.conv_transpose3d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
