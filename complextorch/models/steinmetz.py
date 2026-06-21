r"""
Steinmetz & Analytic Neural Networks
====================================

Architectures that process complex-valued data with **parallel real-valued
subnetworks** whose outputs are coupled into a complex latent (Steinmetz), with
an optional **analytic-signal consistency penalty** (Analytic Neural Network)
that encourages a deterministic, orthogonal relationship between the real and
imaginary channels and provably tightens the generalisation gap.

Reference:

    - **Venkatasubramanian, Pezeshki, Tarokh. Steinmetz Neural Networks for
      Complex-Valued Data.** AISTATS 2025. https://arxiv.org/abs/2409.10075
"""

import torch
import torch.nn as nn

from complextorch.nn.modules.loss import AnalyticSignalLoss

__all__ = ["AnalyticNeuralNetwork", "SteinmetzNetwork"]


class _RealMLP(nn.Module):
    """Plain real-valued MLP (Linear -> ReLU stack)."""

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, depth: int
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = in_features
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(size, hidden_features), nn.ReLU()]
            size = hidden_features
        layers.append(nn.Linear(size, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SteinmetzNetwork(nn.Module):
    r"""
    Steinmetz Network
    -----------------

    Two **parallel real-valued** subnetworks consume the stacked
    ``[Re(z), Im(z)]`` features and produce the real and imaginary parts of a
    complex latent, which are coupled with :func:`torch.complex`:

    .. math::

        u = f_\Re([\Re z, \Im z]), \quad
        v = f_\Im([\Re z, \Im z]), \quad
        \hat{z} = u + j v.

    Args:
        in_features: number of complex input features.
        hidden_features: width of the real subnetworks.
        out_features: number of complex output features.
        depth: number of linear layers in each subnetwork.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.real_branch = _RealMLP(
            2 * in_features, hidden_features, out_features, depth
        )
        self.imag_branch = _RealMLP(
            2 * in_features, hidden_features, out_features, depth
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Map a complex input to a complex output via the coupled real branches.

        Args:
            input (torch.Tensor): complex ``(..., in_features)`` tensor.

        Returns:
            torch.Tensor: complex ``(..., out_features)`` tensor.
        """
        feat = torch.cat([input.real, input.imag], dim=-1)
        return torch.complex(self.real_branch(feat), self.imag_branch(feat))


class AnalyticNeuralNetwork(SteinmetzNetwork):
    r"""
    Analytic Neural Network
    -----------------------

    A :class:`SteinmetzNetwork` paired with the analytic-signal consistency
    penalty (:class:`complextorch.nn.AnalyticSignalLoss`). Add
    :meth:`consistency_loss` of the network output to the task loss during
    training to push the latent towards a true analytic signal
    (:math:`\Im(\hat z) = \mathcal{H}\{\Re(\hat z)\}`), which the paper shows
    lowers the generalisation-gap bound relative to a generic Steinmetz network.

    Args:
        in_features, hidden_features, out_features, depth: see
            :class:`SteinmetzNetwork`.
        consistency_dim: signal dimension for the Hilbert transform in the
            consistency penalty.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int = 2,
        consistency_dim: int = -1,
    ) -> None:
        super().__init__(in_features, hidden_features, out_features, depth)
        self.consistency = AnalyticSignalLoss(dim=consistency_dim)

    def consistency_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Analytic-signal consistency penalty of a (network output) latent."""
        return self.consistency(z)
