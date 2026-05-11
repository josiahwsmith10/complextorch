r"""
Prototype-Distance Classifier Head
==================================

:class:`PrototypeDistance` stores a learnable bank of complex prototypes and
returns negative-distance logits for each prototype. Used as the classifier
head in the CDS reference models (`I`- and `E`-type).

Based on work from the following paper:

    **U. Singhal, Y. Xing, S. X. Yu. Co-Domain Symmetry for Complex-Valued Deep Learning.**

        - CVPR 2022 — ``DistFeatures`` in the reference implementation

        - https://openaccess.thecvf.com/content/CVPR2022/papers/Singhal_Co-Domain_Symmetry_for_Complex-Valued_Deep_Learning_CVPR_2022_paper.pdf
"""

import math
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["PrototypeDistance"]


class PrototypeDistance(nn.Module):
    r"""
    Complex Prototype Distance Classifier
    -------------------------------------

    Holds :math:`K` learnable complex prototypes :math:`p_{c,k}`. For each
    sample ``z`` of shape ``[B, C]``, the logit for class :math:`k` is the
    negative root-mean-squared complex distance to prototype :math:`k`:

    .. math::

        \text{logits}_{b,k}
            = -\tau \cdot \sqrt{\tfrac{1}{C} \sum_{c} |z_{b,c} - p_{c,k}|^2}

    where :math:`\tau \in \mathbb{R}` is a learnable temperature.

    Equivariant ("E-type") use
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    To form a U(1)-equivariant *network*, the prototypes can be pre-rotated by
    a reference complex vector ``y`` (one per sample, per channel) before
    distance:

    .. math::

        p_{b,c,k}^{\prime} = y_{b,c} \cdot p_{c,k}

    Pass ``y`` via the ``reference=`` argument of :meth:`forward`. When the
    network rotates ``z`` by a global :math:`e^{j\psi}`, both ``z`` and ``y``
    rotate identically and the distance is unchanged — so the logits are
    invariant in the I-type call and equivariant-then-invariant in the E-type
    call. This matches ``cds/model.py:225-228``.

    Args:
        in_features: number of complex channels :math:`C`.
        num_prototypes: number of prototypes / output classes :math:`K`.
        temperature_init: initial value of the learnable temperature ``τ``.
    """

    def __init__(
        self,
        in_features: int,
        num_prototypes: int,
        temperature_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_prototypes = num_prototypes
        scale = 1.0 / math.sqrt(in_features)
        proto = torch.empty(in_features, num_prototypes, dtype=torch.cfloat)
        proto.real.normal_(0.0, scale)
        proto.imag.normal_(0.0, scale)
        self.prototypes = nn.Parameter(proto)
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))

    def forward(
        self, input: torch.Tensor, reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(
                f"PrototypeDistance expects input of shape [B, C], got "
                f"{tuple(input.shape)}"
            )
        if not input.is_complex():
            input = input.to(torch.cfloat)

        # prototypes broadcast over batch: [1, C, K]
        proto = self.prototypes.unsqueeze(0)
        if reference is not None:
            if not reference.is_complex():
                reference = reference.to(torch.cfloat)
            if reference.dim() == 1:
                reference = reference.unsqueeze(-1)  # [B] → [B, 1]
            if reference.dim() != 2 or reference.shape[0] != input.shape[0]:
                raise ValueError(
                    f"reference must broadcast against input shape "
                    f"{tuple(input.shape)} on the channel dim; got "
                    f"{tuple(reference.shape)}"
                )
            # Apply rotation: [B, C, 1] * [1, C, K] (broadcasts a scalar reference).
            proto = reference.unsqueeze(-1) * proto

        # diff: [B, C, K]
        diff = input.unsqueeze(-1) - proto
        # Mean over channels of squared absolute difference, then sqrt.
        dist_sq = (diff.real * diff.real + diff.imag * diff.imag).mean(dim=1)
        dist = torch.sqrt(dist_sq.clamp(min=0.0))
        return -dist * self.temperature

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, num_prototypes={self.num_prototypes}"
