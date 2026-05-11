from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t

__all__ = ["wFMConv1d", "wFMConv2d", "wFMReLU", "wFMDistanceLinear"]


def _normalize_weights_squared(weights: torch.Tensor) -> torch.Tensor:
    r"""Normalizes the square of input tensor (weights) such that the sum of the output is 1.
    Follows the function `weightNormalize1` from https://github.com/xingyifei2016/RotLieNet/blob/master/layers.py.

    Args:
        weights (torch.Tensor): input tensor

    Returns:
        torch.Tensor: normalized output
    """
    return (weights**2) / torch.sum(weights**2)


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    r"""Normalizes the input tensor by the sum of its square.
    Follows the function `weightNormalize2` from https://github.com/xingyifei2016/RotLieNet/blob/master/layers.py.

    Args:
        weights (torch.Tensor): input tensor

    Returns:
        torch.Tensor: normalized output
    """
    return weights / torch.sum(weights**2)


def _normalize_rows(weights: torch.Tensor) -> torch.Tensor:
    r"""Normalizes the square of input tensor by each row such that the sum of each row of the output is 1.
    Follows the function `weightNormalize` from https://github.com/xingyifei2016/RotLieNet/blob/master/layers.py.

    Args:
        weights (torch.Tensor): input tensor

    Returns:
        torch.Tensor: normalized output
    """
    return (weights**2) / torch.sum(weights**2, dim=1, keepdim=True)


class _wFMConv2dHelper(nn.Module):
    r"""
    Helper Class for wFMConv2d
    ----------------------------
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = (1, 1),
        padding: _size_2_t = (0, 0),
        weight_dropout: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        super(_wFMConv2dHelper, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps

        prod_kernel_size = np.prod(kernel_size)

        self.dropout = nn.Dropout(weight_dropout)

        # Weight matrices
        self.weight_matrix_ang1 = nn.Parameter(
            torch.rand(in_channels, prod_kernel_size), requires_grad=True
        )

        self.weight_matrix_ang2 = nn.Parameter(
            torch.rand(out_channels, in_channels), requires_grad=True
        )

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

    def compute_output_shape(self, input_shape) -> Tuple[int]:
        return tuple(
            int(np.floor((in_shape + 2 * padding - (kernel_size - 1) - 1) / stride + 1))
            for in_shape, padding, kernel_size, stride in zip(
                input_shape, self.padding, self.kernel_size, self.stride
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, mag_ang, in_channels, *input_shape = input.shape

        assert mag_ang == 2, "Input must be complex valued in polar form (mag, ang)"
        assert in_channels == self.in_channels, "Input channels must match"

        out_channels = self.out_channels
        kernel_size = self.kernel_size
        prod_kernel_size = np.prod(kernel_size)

        output_shape = self.compute_output_shape(input_shape)
        L = np.prod(output_shape)  # Total number of unfolded blocks

        input = input.view(batch_size * 2, in_channels, *input_shape)

        # unfolded shape: (batch_size * 2, in_channels * prod_kernel_size, L)
        temporal_buckets = self.unfold(input).view(
            batch_size, 2, in_channels, prod_kernel_size, L
        )

        ### Do magnitude processing
        tb_mag = torch.log(
            temporal_buckets[:, 0]
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * L, in_channels, prod_kernel_size)
            + self.eps
        )

        # Normalize the weights
        wmm1 = _normalize_rows(self.dropout(self.weight_matrix_ang1))
        wmm2 = _normalize_rows(self.dropout(self.weight_matrix_ang2))

        out_mag = (
            torch.sum(tb_mag * wmm1, dim=2).unsqueeze(1).repeat(1, out_channels, 1)
        )

        out_mag = torch.exp(
            torch.sum(out_mag * wmm2, dim=2)
            .view(batch_size, 1, *output_shape, out_channels)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )

        ### Do phase processing
        tb_ang = (
            temporal_buckets[:, 1]
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * L, in_channels, prod_kernel_size)
        )

        # Normalize the weights
        wma1 = _normalize_weights_squared(self.weight_matrix_ang1)
        wma2 = _normalize_weights_squared(self.weight_matrix_ang2)

        out_ang = (
            torch.sum(tb_ang * wma1, dim=2).unsqueeze(1).repeat(1, out_channels, 1)
        )

        out_ang = (
            torch.sum(out_ang * wma2, dim=2)
            .view(batch_size, 1, *output_shape, out_channels)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )

        return torch.cat((out_mag, out_ang), dim=1)


class wFMConv2d(nn.Module):
    r"""
    2-D Weighted Frechet Mean Convolution Layer
    ---------------------------------------------

    In a paper title `Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold`, the authors R Chakraborty, Y Xing, and S Yu introduce a complex-valued convolution operator offering similar equivariance properties to the spatial equivariance of the traditional real-valued convolution operator.
    By approach the complex domain as a Riemannian homogeneous space consisting of the product of planar rotation and non-zero scaling, they define a convolution operator equivariant to phase shift and amplitude scaling.
    Although their paper shows promising results in reducing the number of parameters of a complex-valued network for several problems, their work has not gained mainstream support.

    As the authors mention in the final bullet point in Section IV-A1,

        If :math:`d` is the manifold distance in (2) for the Euclidean
        space that is also Riemannian, then wFM has exactly the
        weighted average as its closed-form solution. That is, our
        wFM convolution on the Euclidean manifold is reduced
        to the standard convolution, although with the additional
        convexity constraint on the weights.

    Hence, the implementation closely follows the conventional convolution operator with the exception of the weight normalization.

    Note: the weight normalization, although consistent with the authors' implementation, lacks adequate explanation from the literature and could be improved for further clarity.

    Based on work from the following paper:

        **R Chakraborty, Y Xing, S Yu. SurReal: Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold**

            - Eqs. (14)-(16)

            - https://arxiv.org/abs/1910.11334

            - Modified from implementation: https://github.com/xingyifei2016/RotLieNet (yields consistent results as this implementation)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = (1, 1),
        padding: _size_2_t = (0, 0),
        weight_dropout: float = 0.0,
    ) -> None:
        super(wFMConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_dropout = weight_dropout

        prod_kernel_size = np.prod(kernel_size)

        # Weight matrices for magnitude and angle
        self.weight_matrix_mag = nn.Parameter(
            torch.rand(in_channels, prod_kernel_size), requires_grad=True
        )

        self.weight_matrix_ang = nn.Parameter(
            torch.rand(in_channels, prod_kernel_size), requires_grad=True
        )

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

        self.wFM_conv = _wFMConv2dHelper(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_dropout=weight_dropout,
        )

        # Lazily built and cached by input spatial shape so we don't reallocate
        # an ``nn.Fold`` on every forward when shapes are stable.
        self._fold_cache: "dict[tuple, nn.Fold]" = {}

    def _get_fold(self, input_shape) -> nn.Fold:
        key = tuple(input_shape)
        fold = self._fold_cache.get(key)
        if fold is None:
            fold = nn.Fold(
                output_size=input_shape,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            self._fold_cache[key] = fold
        return fold

    def compute_output_shape(self, input_shape) -> Tuple[int]:
        return tuple(
            int(np.floor((in_shape + 2 * padding - (kernel_size - 1) - 1) / stride + 1))
            for in_shape, padding, kernel_size, stride in zip(
                input_shape, self.padding, self.kernel_size, self.stride
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes the 2-D weighted Frechet mean (wFM) convolution.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        batch_size, in_channels, *input_shape = input.shape

        assert in_channels == self.in_channels, "Input channels must match"

        kernel_size = self.kernel_size
        prod_kernel_size = np.prod(kernel_size)

        output_shape = self.compute_output_shape(input_shape)
        L = np.prod(output_shape)  # Total number of unfolded blocks

        fold = self._get_fold(input_shape)

        # Separate magnitude and angle from torch.Tensor input
        x_mag, x_ang = input.abs(), input.angle()

        ### Do magnitude processing
        x_mag = self.unfold(x_mag).view(batch_size, in_channels, prod_kernel_size, L)

        x_mag = (
            x_mag.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * L, in_channels, prod_kernel_size)
        )

        x_mag = x_mag + _normalize_weights_squared(self.weight_matrix_mag)

        x_mag = (
            x_mag.view(batch_size, *output_shape, in_channels * prod_kernel_size)
            .permute(0, 3, 1, 2)
            .contiguous()
            .unsqueeze(1)
        )

        ### Do phase processing
        x_ang = self.unfold(x_ang).view(batch_size, in_channels, prod_kernel_size, L)

        x_ang = (
            x_ang.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * L, in_channels, prod_kernel_size)
        )

        x_ang = x_ang * _normalize_weights(self.weight_matrix_ang)

        x_ang = (
            x_ang.view(batch_size, *output_shape, in_channels * prod_kernel_size)
            .permute(0, 3, 1, 2)
            .contiguous()
            .unsqueeze(1)
        )

        # Stack the magnitude and phase tensors
        in_fold = fold(
            torch.cat((x_mag, x_ang), dim=1).view(
                batch_size, 2 * in_channels * prod_kernel_size, L
            )
        ).view(batch_size, 2, in_channels, *input_shape)

        x_out = self.wFM_conv(in_fold)
        return torch.polar(x_out[:, 0], x_out[:, 1])


class wFMConv1d(nn.Module):
    r"""
    1-D Weighted Frechet Mean Convolution Layer
    ---------------------------------------------

    In a paper title `Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold`, the authors R Chakraborty, Y Xing, and S Yu introduce a complex-valued convolution operator offering similar equivariance properties to the spatial equivariance of the traditional real-valued convolution operator.
    By approach the complex domain as a Riemannian homogeneous space consisting of the product of planar rotation and non-zero scaling, they define a convolution operator equivariant to phase shift and amplitude scaling.
    Although their paper shows promising results in reducing the number of parameters of a complex-valued network for several problems, their work has not gained mainstream support.

    As the authors mention in the final bullet point in Section IV-A1,

        If :math:`d` is the manifold distance in (2) for the Euclidean
        space that is also Riemannian, then wFM has exactly the
        weighted average as its closed-form solution. That is, our
        wFM convolution on the Euclidean manifold is reduced
        to the standard convolution, although with the additional
        convexity constraint on the weights.

    Hence, the implementation closely follows the conventional convolution operator with the exception of the weight normalization.

    Note: the weight normalization, although consistent with the authors' implementation, lacks adequate explanation from the literature and could be improved for further clarity.

    Note: This is a wrapper around wFMConv2d that performs a 1D convolution

    Based on work from the following paper:

        **R Chakraborty, Y Xing, S Yu. SurReal: Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold**

            - Eqs. (14)-(16)

            - https://arxiv.org/abs/1910.11334

            - Modified from implementation: https://github.com/xingyifei2016/RotLieNet (yields consistent results as this implementation)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        weight_dropout: float = 0.0,
    ) -> None:
        super(wFMConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_dropout = weight_dropout

        self.conv1d = wFMConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            weight_dropout=weight_dropout,
        )

        self.wFM_conv = self.conv1d.wFM_conv

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Computes the 1-D weighted Frechet mean (wFM) convolution. See :class:`wFMConv2d` for more implementation details as :class:`wFMConv1d` is a wrapper around :class:`wFMConv2d`.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.conv1d(input.unsqueeze(-2)).squeeze(-2)

    @property
    def weight_matrix_ang(self) -> torch.Tensor:
        return self.conv1d.weight_matrix_ang

    @property
    def weight_matrix_mag(self) -> torch.Tensor:
        return self.conv1d.weight_matrix_mag


class wFMReLU(nn.Module):
    r"""
    Weighted Fréchet Mean ReLU
    ----------------------------

    Manifold-aware nonlinearity that complements :class:`wFMConv1d` /
    :class:`wFMConv2d`. Performs an additive shift on the magnitude (in
    log-domain semantics) and a multiplicative scaling on the phase, with
    weight normalisations chosen so that the operation lies on the
    rotation+scaling manifold:

    .. math::

        \begin{aligned}
            \tilde{|z|}_c &= |z|_c + \frac{(w^{m}_c)^2}{\sum_k (w^{m}_k)^2} \\
            \tilde{\theta}_c &= \arg(z_c) \cdot \frac{w^{\theta}_c}{\sum_k (w^{\theta}_k)^2} \\
            y_c &= \tilde{|z|}_c \cdot e^{j \tilde{\theta}_c}
        \end{aligned}

    Both :math:`w^m, w^\theta \in \mathbb{R}^C` are learnable. The normalisations
    are exactly those used by :class:`wFMConv2d` (``weightNormalize1`` /
    ``weightNormalize2`` in the reference implementation).

    Based on work from the following paper:

        **R Chakraborty, Y Xing, S Yu. SurReal: Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold**

            - https://arxiv.org/abs/1910.11334

            - Reference implementation: `manifoldReLUv2angle` in https://github.com/xingyifei2016/RotLieNet
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight_phase = nn.Parameter(torch.empty(num_channels).uniform_(0.0, 1.0))
        self.weight_mag = nn.Parameter(torch.empty(num_channels).uniform_(0.0, 1.0))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            input = input.to(torch.cfloat)
        # Broadcast (C,) → (1, C, 1, ...) for an N-D input.
        view_shape = [1] * input.dim()
        view_shape[1] = self.num_channels
        w_phase = _normalize_weights(self.weight_phase + self.eps).view(*view_shape)
        w_mag = _normalize_weights_squared(self.weight_mag + self.eps).view(*view_shape)
        mag = input.abs() + w_mag
        phase = input.angle() * w_phase
        return torch.polar(mag, phase)

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}"


class wFMDistanceLinear(nn.Module):
    r"""
    Weighted Fréchet Mean Distance Linear Head
    --------------------------------------------

    Computes a single weighted Fréchet mean :math:`M` over every element of a
    complex input, then returns a **real-valued** distance map combining
    rotation distance (on phase) and a log-magnitude ratio. Suitable as a
    classification head where the goal is to summarise the input's deviation
    from a learned manifold center.

    For a flattened input :math:`z \in \mathbb{C}^N`,

    .. math::

        \begin{aligned}
            \bar{w}_n &= \frac{(w_n)^2}{\sum_k (w_k)^2} \\
            M_\theta &= \tanh(-b_\theta) \cdot \sum_n \arg(z_n) \cdot \bar{w}_n \\
            M_r &= \exp(-b_r^2) + \exp\Bigl(\sum_n \log(|z_n| + \varepsilon) \cdot \bar{w}_n\Bigr) \\
            d_n &= w_\theta^2 \cdot |\arg(z_n) - M_\theta|
                  + w_r^2 \cdot |\log(|z_n| / (M_r + \varepsilon))|
        \end{aligned}

    The output is real-valued and has the same shape as the input (with the
    leading complex axis flattened to the same shape).

    .. note::
        Unlike :class:`complextorch.nn.Linear` this layer returns a real-valued
        tensor (it produces invariants for classification, not complex
        features). The "Distance" suffix is the reminder.

    Based on work from the following paper:

        **R Chakraborty, Y Xing, S Yu. SurReal: Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold**

            - https://arxiv.org/abs/1910.11334

            - Reference implementation: ``ComplexLinearangle2Dmw_outfield`` in https://github.com/xingyifei2016/RotLieNet
    """

    def __init__(self, input_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        # Per-element wFM weights and a pair of (phase, magnitude) combination
        # weights / biases.
        self.weights = nn.Parameter(torch.empty(input_dim).uniform_(0.0, 1.0))
        self.combine_weight = nn.Parameter(torch.empty(2).uniform_(0.0, 1.0))
        self.bias = nn.Parameter(torch.empty(2).uniform_(0.0, 1.0))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            input = input.to(torch.cfloat)
        original_shape = input.shape
        batch_size = original_shape[0]
        flat = input.reshape(batch_size, -1)
        if flat.shape[1] != self.input_dim:
            raise ValueError(
                f"wFMDistanceLinear expects flattened input of size "
                f"{self.input_dim}, got {flat.shape[1]}"
            )
        phase = flat.angle()
        mag = flat.abs()

        w = _normalize_weights_squared(self.weights)  # [N], sums to 1
        M_phase = (phase * w).sum(dim=1) * torch.tanh(-self.bias[0])  # [B]
        log_mag = torch.log(mag + self.eps)
        M_mag = torch.exp((log_mag * w).sum(dim=1)) + torch.exp(
            -self.bias[1] ** 2
        )  # [B]

        dist_phase = (phase - M_phase.unsqueeze(1)).abs()  # [B, N]
        dist_mag = torch.log(mag / (M_mag.unsqueeze(1) + self.eps)).abs()  # [B, N]
        dist = (
            self.combine_weight[0] ** 2 * dist_phase
            + self.combine_weight[1] ** 2 * dist_mag
        )
        return dist.reshape(original_shape)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}"
