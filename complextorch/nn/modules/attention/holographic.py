r"""
Holographic (Interference-Aware) Complex Attention
==================================================

A complex-valued self-attention variant that treats attention as physical wave
interference rather than a phase-blind correlation. It (i) gates the attention
logits by the query/key **phase discrepancy** and (ii) performs a **coherent
superposition** of the values -- rotating each value by its query/key phase
difference before the weighted sum, so aligned phases add constructively and
opposed phases partially cancel.

Based on work from the following paper:

    **Holographic Transformers for Complex-Valued Signal Processing: Integrating
    Phase Interference into Self-Attention.**

        - https://arxiv.org/abs/2509.19331

For a token pair :math:`(i, j)` with complex score
:math:`s_{ij} = Q_i K_j^H` and phase difference
:math:`\Delta\phi_{ij} = \angle s_{ij}`:

.. math::

    \text{sim}_{ij} &= \frac{\Re(s_{ij})}{\lVert Q_i\rVert\,\lVert K_j\rVert + \epsilon}, \\
    W_{ij} &= \frac{\text{sim}_{ij}}{\sqrt{d_k}}\, e^{-\alpha\lvert\Delta\phi_{ij}\rvert}, \qquad
    a_{ij} = \texttt{SoftMax}_j(W_{ij}), \\
    H_i &= \sum_j a_{ij}\, V_j\, e^{j\Delta\phi_{ij}}.

The phase-discrepancy weight :math:`\alpha \ge 0` is learnable (kept
non-negative via :func:`~torch.nn.functional.softplus`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HolographicAttention"]


class HolographicAttention(nn.Module):
    r"""
    Holographic / Interference-Aware Attention
    ------------------------------------------

    Drop-in sibling of
    :class:`~complextorch.nn.ScaledDotProductAttention` with the same
    ``forward(q, k, v)`` signature; usable directly or selected inside
    :class:`~complextorch.nn.MultiheadAttention` via ``attention='holographic'``.

    Args:
        temperature: score temperature, typically :math:`\sqrt{d_k}`.
        attn_dropout: dropout probability applied to the (real) attention
            weights.
        alpha_init: initial value of the phase-discrepancy weight :math:`\alpha`.
        eps: small constant guarding the magnitude normalisation.
    """

    def __init__(
        self,
        temperature: float,
        attn_dropout: float = 0.1,
        alpha_init: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
        # Stored as a raw parameter; softplus keeps the effective weight >= 0.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        r"""Computes interference-aware attention.

        Args:
            q (torch.Tensor): complex-valued query tensor ``(..., L_q, d_k)``.
            k (torch.Tensor): complex-valued key tensor ``(..., L_k, d_k)``.
            v (torch.Tensor): complex-valued value tensor ``(..., L_k, d_v)``.

        Returns:
            torch.Tensor: coherently-superposed complex output ``(..., L_q, d_v)``.
        """
        # Hermitian inner product -> complex score; its phase is Delta-phi.
        scores = torch.matmul(q, k.conj().transpose(-2, -1))
        dphi = scores.angle()

        q_norm = torch.linalg.vector_norm(q, dim=-1).unsqueeze(-1)
        k_norm = torch.linalg.vector_norm(k, dim=-1).unsqueeze(-2)
        sim = scores.real / (q_norm * k_norm + self.eps)

        gate = torch.exp(-F.softplus(self.alpha) * dphi.abs())
        weights = self.softmax(sim / self.temperature * gate)
        weights = self.dropout(weights)

        # Coherent superposition: rotate each value by its phase offset.
        coeff = weights * torch.polar(torch.ones_like(dphi), dphi)
        return torch.matmul(coeff, v)
