r"""
Learnable Phase / Complex-Scaling Modules
=========================================

- :class:`PhaseShift` multiplies its input by :math:`e^{j\phi}` with a learnable
  phase :math:`\phi` (magnitude fixed to 1).
- :class:`ComplexScaling` multiplies its input by the general complex scalar
  :math:`\alpha + j\beta` with both real and imaginary parts learnable
  (magnitude and phase both learnable).
"""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn

__all__ = ["PhaseShift", "ComplexScaling"]


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


class ComplexScaling(nn.Module):
    r"""
    Learnable Complex Scaling
    -------------------------

    Multiplies a complex input by the learnable complex scalar
    :math:`\alpha + j\beta`:

    .. math::

        y = (\alpha + j\beta) \cdot z
          = (\alpha \Re z - \beta \Im z) + j(\beta \Re z + \alpha \Im z)

    Unlike :class:`PhaseShift` (which restricts the multiplier to unit
    magnitude), :class:`ComplexScaling` learns both magnitude and phase.

    Broadcasting matches :class:`PhaseShift`: pass an int / tuple of ints for
    ``num_features`` and the parameter shape is identical to it. When
    ``num_features`` is a single int, the parameter has shape ``(num_features,)``
    and is broadcast at ``broadcast_dim`` (default 1, i.e. the channel axis of a
    ``(B, C, ...)`` input).

    Based on work from the following paper:

        **U. Singhal, Y. Xing, S. X. Yu. Co-Domain Symmetry for Complex-Valued Deep Learning.**

            - CVPR 2022 — `scaling_layer` in the reference implementation

            - https://openaccess.thecvf.com/content/CVPR2022/papers/Singhal_Co-Domain_Symmetry_for_Complex-Valued_Deep_Learning_CVPR_2022_paper.pdf

    Args:
        num_features: shape of the learnable scale parameters. ``()`` or ``1``
            gives a single scalar scale.
        broadcast_dim: when ``num_features`` is an int, the parameter is
            broadcast to the input at this dim. Use ``1`` (default) for
            ``(B, C, ...)`` and ``-1`` for ``(B, ..., C)``.
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
        # Matches the reference (cds/layers.py:474-475): uniform [0, 1).
        self.alpha = nn.Parameter(torch.empty(shape).uniform_(0.0, 1.0))
        self.beta = nn.Parameter(torch.empty(shape).uniform_(0.0, 1.0))

    def _broadcast(self, t: torch.Tensor, input_dim: int) -> torch.Tensor:
        if t.dim() == 1 and input_dim > 1:
            dim = (
                self.broadcast_dim
                if self.broadcast_dim >= 0
                else input_dim + self.broadcast_dim
            )
            shape = [1] * input_dim
            shape[dim] = t.shape[0]
            return t.view(*shape)
        return t

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_complex():
            input = input.to(torch.cfloat)
        alpha = self._broadcast(self.alpha, input.dim())
        beta = self._broadcast(self.beta, input.dim())
        scale = torch.complex(alpha, beta)
        return input * scale

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, broadcast_dim={self.broadcast_dim}"
