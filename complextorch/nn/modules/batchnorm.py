import torch
import torch.nn as nn
from torch.nn import init

from .. import functional as cvF

__all__ = ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d"]


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

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
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
    Similar to the `PyTorch BatchNorm1d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>`_ implementation but performs the proper batch normalization for complex-valued data.

    See `torch.nn.BatchNorm1d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>`_ for additional details.

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
    Similar to the `PyTorch BatchNorm2d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_ implementation but performs the proper batch normalization for complex-valued data.

    See `torch.nn.BatchNorm2d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_ for additional details.

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
    Similar to the `PyTorch BatchNorm3d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html>`_ implementation but performs the proper batch normalization for complex-valued data.

    See `torch.nn.BatchNorm3d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html>`_ for additional details.

    Based on work from the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**

            - Section 6

            - https://arxiv.org/abs/2302.08286
    """

    def _check_input_dim(self, input: torch.Tensor) -> None:
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")
