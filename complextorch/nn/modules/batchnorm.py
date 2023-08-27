import torch
import torch.nn as nn
from torch.nn import init

from .. import functional as cvF
from ... import CVTensor

__all__ = ['CVBatchNorm1d', 'CVBatchNorm2d', 'CVBatchNorm3d']


class _CVBatchNorm(nn.Module):
    """
    The base class for Complex-valeud batch normalization layer.

    Taken from 'torch.nn.modules.batchnorm.'
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

    def forward(self, input: CVTensor) -> CVTensor:
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

        return cvF.cv_batch_norm(
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


class CVBatchNorm1d(_CVBatchNorm):
    """
    Complex-valued batch normalization for 2D or 3D data.

    See torch.nn.BatchNorm1d for details.
    """

    def _check_input_dim(self, input: CVTensor) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class CVBatchNorm2d(_CVBatchNorm):
    """
    Complex-valued batch normalization for 4D data.

    See torch.nn.BatchNorm2d for details.
    """

    def _check_input_dim(self, input: CVTensor) -> None:
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class CVBatchNorm3d(_CVBatchNorm):
    """
    Complex-valued batch normalization for 5D data.

    See torch.nn.BatchNorm3d for details.
    """

    def _check_input_dim(self, input: CVTensor) -> None:
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")
