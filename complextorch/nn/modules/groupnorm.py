r"""
Complex-Valued GroupNorm
========================

Group normalization adapted to complex tensors. Splits channels into
``num_groups`` groups, applies 2x2 whitening within each group, then a
per-channel 2x2 affine transform + 2-vector bias. No running statistics
(differs from :class:`BatchNorm2d`).
"""

import torch
import torch.nn as nn
from torch.nn import init

from complextorch.nn.functional import inv_sqrtm2x2

__all__ = ["GroupNorm"]


class GroupNorm(nn.Module):
    r"""
    Complex-Valued Group Normalization
    ----------------------------------

    Like :class:`torch.nn.GroupNorm`, but applies the Trabelsi 2x2 whitening
    transform within each group, then a per-channel 2x2 affine.

    Args:
        num_groups: number of groups to divide the channels into; must divide
            ``num_channels``.
        num_channels: number of channels in the input.
        eps: numerical stabilizer.
        affine: if ``True``, applies a learnable per-channel 2x2 affine +
            2-vector bias.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.empty(2, 2, num_channels))
            self.bias = nn.Parameter(torch.empty(2, num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.affine:
            return
        self.weight.data.copy_(0.70710678118 * torch.eye(2, 2).unsqueeze(-1))
        init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            raise TypeError(
                f"GroupNorm expects a complex input, got dtype={input.dtype}"
            )
        b, c = input.shape[:2]
        if c != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {c}")
        spatial = input.shape[2:]
        g = self.num_groups
        c_per_g = c // g

        # Reshape to (B, G, C//G, *spatial). Whiten over (C//G, *spatial) per group.
        re = input.real.view(b, g, c_per_g, *spatial)
        im = input.imag.view(b, g, c_per_g, *spatial)

        # Axes to reduce over within each group: c_per_g + spatial dims.
        reduce_axes = tuple(range(2, 2 + 1 + len(spatial)))  # dim 2 onwards
        # Mean per (B, G, 1, 1, ...) — center
        mean_r = re.mean(dim=reduce_axes, keepdim=True)
        mean_i = im.mean(dim=reduce_axes, keepdim=True)
        re_c = re - mean_r
        im_c = im - mean_i

        # 2x2 covariance per group
        v_rr = (re_c * re_c).mean(dim=reduce_axes) + self.eps
        v_ii = (im_c * im_c).mean(dim=reduce_axes) + self.eps
        v_ir = (re_c * im_c).mean(dim=reduce_axes)

        p, q, _, s = inv_sqrtm2x2(v_rr, v_ir, None, v_ii, symmetric=True)

        # Broadcast p, q, s back over the per-group reduced shape: (B, G, 1, 1, ...)
        bcast_shape = (b, g) + (1,) * (1 + len(spatial))
        p = p.view(bcast_shape)
        q = q.view(bcast_shape)
        s = s.view(bcast_shape)

        out_r = p * re_c + q * im_c
        out_i = q * re_c + s * im_c

        # Flatten group dim back: (B, C, *spatial)
        out_r = out_r.reshape(b, c, *spatial)
        out_i = out_i.reshape(b, c, *spatial)

        if self.affine:
            # weight has shape (2, 2, C); broadcast over batch + spatial
            chan_shape = (1, c) + (1,) * len(spatial)
            w = self.weight
            new_r = w[0, 0].view(chan_shape) * out_r + w[0, 1].view(chan_shape) * out_i
            new_i = w[1, 0].view(chan_shape) * out_r + w[1, 1].view(chan_shape) * out_i
            new_r = new_r + self.bias[0].view(chan_shape)
            new_i = new_i + self.bias[1].view(chan_shape)
            out_r, out_i = new_r, new_i

        return torch.complex(out_r, out_i)

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, num_channels={self.num_channels}, "
            f"eps={self.eps}, affine={self.affine}"
        )
