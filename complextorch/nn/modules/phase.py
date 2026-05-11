r"""
Learnable Phase-Rotation Module
===============================

Implements :class:`PhaseShift`, a complex-valued layer that multiplies its
input by ``exp(j * phi)`` with a learnable phase ``phi`` (broadcasting follows
PyTorch conventions).
"""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn

__all__ = ["PhaseShift"]


class PhaseShift(nn.Module):
    r"""
    Learnable Phase Shift
    ---------------------

    Multiplies a complex input by :math:`e^{j\phi}`, where :math:`\phi` is a
    learnable parameter:

    .. math::

        y = x \cdot e^{j\phi}

    The shape of :math:`\phi` is given by ``num_features``; PyTorch
    broadcasting rules apply between the input and ``exp(j*phi)``. Pass a
    scalar (``()``) for a single global rotation, ``(C,)`` for a per-channel
    rotation, or a higher-rank tuple for finer control.

    Args:
        num_features: shape of the learnable phase tensor. ``()`` or ``1``
            gives a scalar rotation.
        broadcast_dim: when ``num_features`` is an int, ``phi`` is created with
            shape ``(num_features,)``. To use it as the channel dim of a
            ``(B, C, ...)`` input, set ``broadcast_dim=1`` (default). For a
            ``(B, ..., C)`` layout set ``broadcast_dim=-1``.
    """

    def __init__(
        self,
        num_features: Union[int, Tuple[int, ...]] = 1,
        broadcast_dim: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(num_features, int):
            shape: Tuple[int, ...] = (num_features,) if num_features != 1 else ()
        else:
            shape = tuple(num_features)
        self.num_features = num_features
        self.broadcast_dim = broadcast_dim
        # Initialize phases uniformly in [-pi, pi]
        phi = torch.empty(shape)
        with torch.no_grad():
            phi.uniform_(-math.pi, math.pi)
        self.phi = nn.Parameter(phi)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reshape phi to broadcast over input. For a 1-D phi of shape (C,) the
        # default reshape places it at dim broadcast_dim of the input.
        phi = self.phi
        if phi.dim() == 1 and input.dim() > 1:
            ndim = input.dim()
            dim = (
                self.broadcast_dim
                if self.broadcast_dim >= 0
                else ndim + self.broadcast_dim
            )
            shape = [1] * ndim
            shape[dim] = phi.shape[0]
            phi = phi.view(*shape)
        rotor = torch.polar(torch.ones_like(phi), phi)
        if not input.is_complex():
            input = input.to(torch.cfloat)
        return input * rotor

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, broadcast_dim={self.broadcast_dim}"
