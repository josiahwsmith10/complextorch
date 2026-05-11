import torch
import torch.nn as nn
from torch.nn import init

from complextorch.nn import functional as cvF

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "MagBatchNorm1d",
    "MagBatchNorm2d",
    "MagBatchNorm3d",
    "NaiveBatchNorm1d",
    "NaiveBatchNorm2d",
    "NaiveBatchNorm3d",
]


class _BatchNorm(nn.Module):
    r"""
    Complex-Valued Batch Normalization Base Class
    ---------------------------------------------

    Closely replicated from torch.nn.modules.batchnorm.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(2, 2, num_features))
            self.bias = torch.nn.Parameter(torch.empty(2, num_features))

        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.empty(2, num_features))
            self.register_buffer("running_var", torch.empty(2, 2, num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )

        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

        self.reset_running_stats()
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if not self.track_running_stats:
            return

        self.num_batches_tracked.zero_()

        self.running_mean.zero_()
        self.running_var.copy_(torch.eye(2, 2).unsqueeze(-1))

    def reset_parameters(self) -> None:
        if not self.affine:
            return

        self.weight.data.copy_(0.70710678118 * torch.eye(2, 2).unsqueeze(-1))
        init.zeros_(self.bias)

    def _check_input_dim(self, input) -> None:
        raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        exponential_average_factor = 0.0 if self.momentum is None else self.momentum

        if self.training and self.track_running_stats:  # noqa: SIM102 — kept nested to mirror torch.nn.modules.batchnorm._BatchNorm.forward
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return cvF.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

    def extra_repr(self) -> None:
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**vars(self))
        )


class BatchNorm1d(_BatchNorm):
    r"""
    1-D Complex-Valued Batch Normalization
    --------------------------------------

    Complex-valued batch normalization for 2-D and 3-D tensors.
    Similar to the PyTorch :class:`torch.nn.BatchNorm1d` implementation but performs the proper batch normalization for complex-valued data.

    See :class:`torch.nn.BatchNorm1d` for additional details.

    Based on work from the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Section 6

            - https://arxiv.org/abs/2302.08286
    """

    def _check_input_dim(self, input: torch.Tensor) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class BatchNorm2d(_BatchNorm):
    r"""
    2-D Complex-Valued Batch Normalization
    --------------------------------------

    Complex-valued batch normalization for 4-D tensors.
    Similar to the PyTorch :class:`torch.nn.BatchNorm2d` implementation but performs the proper batch normalization for complex-valued data.

    See :class:`torch.nn.BatchNorm2d` for additional details.

    Based on work from the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Section 6

            - https://arxiv.org/abs/2302.08286
    """

    def _check_input_dim(self, input: torch.Tensor) -> None:
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class BatchNorm3d(_BatchNorm):
    r"""
    3-D Complex-Valued Batch Normalization
    --------------------------------------

    Complex-valued batch normalization for 5-D tensors.
    Similar to the PyTorch :class:`torch.nn.BatchNorm3d` implementation but performs the proper batch normalization for complex-valued data.

    See :class:`torch.nn.BatchNorm3d` for additional details.

    Based on work from the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Section 6

            - https://arxiv.org/abs/2302.08286
    """

    def _check_input_dim(self, input: torch.Tensor) -> None:
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")


class _NaiveBatchNorm(nn.Module):
    r"""
    Naive (split) Complex Batch Normalization Base
    ----------------------------------------------

    Applies an independent :class:`torch.nn.BatchNorm{1,2,3}d` to the real and
    imaginary parts of the input. Cheaper than the Trabelsi 2×2-whitening
    :class:`_BatchNorm` (about half the cost) but does not decorrelate the
    real/imag components. Useful as a baseline.
    """

    _real_bn_class = nn.BatchNorm1d  # overridden per dim

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.bn_r = self._real_bn_class(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.bn_i = self._real_bn_class(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.bn_r(input.real), self.bn_i(input.imag))

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"


class NaiveBatchNorm1d(_NaiveBatchNorm):
    r"""1-D split-form complex BatchNorm. See :class:`_NaiveBatchNorm`."""

    _real_bn_class = nn.BatchNorm1d


class NaiveBatchNorm2d(_NaiveBatchNorm):
    r"""2-D split-form complex BatchNorm. See :class:`_NaiveBatchNorm`."""

    _real_bn_class = nn.BatchNorm2d


class NaiveBatchNorm3d(_NaiveBatchNorm):
    r"""3-D split-form complex BatchNorm. See :class:`_NaiveBatchNorm`."""

    _real_bn_class = nn.BatchNorm3d


class _MagBatchNorm(nn.Module):
    r"""
    Magnitude-Only Complex Batch Normalization Base
    -----------------------------------------------

    Applies an ordinary real-valued :class:`torch.nn.BatchNorm{1,2,3}d` to the
    magnitude :math:`|z|` and rescales :math:`z` to match:

    .. math::

        y = z \cdot \frac{\operatorname{BN}(|z|)}{|z| + \varepsilon}

    The output's phase is identical to the input's, so the operator is
    **U(1)-equivariant**: rotating the input by :math:`e^{j\psi}` rotates the
    output by exactly the same angle. This is distinct from the standard
    :class:`BatchNorm{1,2,3}d` (Trabelsi 2×2 whitening), which decorrelates the
    real/imag covariance but is *not* phase-equivariant.

    Running statistics, affine parameters, and ``eps``/``momentum`` semantics
    follow :class:`torch.nn.BatchNorm` directly — the underlying real BN is
    stored as ``self.bn`` so its ``state_dict`` is portable.

    Based on work from the following paper:

        **U. Singhal, Y. Xing, S. X. Yu. Co-Domain Symmetry for Complex-Valued Deep Learning.**

            - CVPR 2022 — `VNCBN` ("Vector-Norm Complex Batch Norm") in the reference implementation

            - https://openaccess.thecvf.com/content/CVPR2022/papers/Singhal_Co-Domain_Symmetry_for_Complex-Valued_Deep_Learning_CVPR_2022_paper.pdf
    """

    _real_bn_class = nn.BatchNorm1d  # overridden per dim

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.bn = self._real_bn_class(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_complex():
            magnitude = input.abs()
            normalized = self.bn(magnitude)
            scale = normalized / (magnitude + self.eps)
            return input * scale.to(input.dtype)
        return self.bn(input)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"


class MagBatchNorm1d(_MagBatchNorm):
    r"""1-D magnitude-only complex BatchNorm. See :class:`_MagBatchNorm`."""

    _real_bn_class = nn.BatchNorm1d


class MagBatchNorm2d(_MagBatchNorm):
    r"""2-D magnitude-only complex BatchNorm. See :class:`_MagBatchNorm`."""

    _real_bn_class = nn.BatchNorm2d


class MagBatchNorm3d(_MagBatchNorm):
    r"""3-D magnitude-only complex BatchNorm. See :class:`_MagBatchNorm`."""

    _real_bn_class = nn.BatchNorm3d
